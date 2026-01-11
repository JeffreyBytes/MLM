#[cfg(target_family = "unix")]
use std::os::unix::fs::MetadataExt as _;
#[cfg(target_family = "windows")]
use std::os::windows::fs::MetadataExt as _;
use std::{
    collections::BTreeMap,
    fs::{self, File, Metadata},
    io::{BufWriter, ErrorKind, Write},
    ops::Deref,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use file_id::get_file_id;
use log::error;
use mlm_db::{
    ClientStatus, DatabaseExt as _, ErroredTorrentId, Event, EventType, LibraryItem,
    LibraryItemKey, LibraryMismatch, SelectedTorrent, SelectedTorrentKey, SeriesEntries,
    SeriesEntry, Size, Timestamp, Torrent, TorrentMeta,
};
use mlm_mam::{api::MaM, meta::MetaError, search::MaMTorrent};
use mlm_parse::normalize_title;
use native_db::Database;
use once_cell::sync::Lazy;
use qbit::{
    models::{Torrent as QbitTorrent, TorrentContent},
    parameters::TorrentListParams,
};
use regex::Regex;
use tokio::fs::create_dir_all;
use tracing::{Level, debug, instrument, span, trace, warn};

use crate::{
    audiobookshelf::{self as abs},
    autograbber::update_torrent_meta,
    cleaner::remove_library_files,
    config::{Config, Library, LibraryLinkMethod, QbitConfig},
    logging::{TorrentMetaError, update_errored_torrent, write_event},
};

pub static DISK_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?:CD|Disc|Disk)\s*(\d+)").unwrap());

struct ItemFile {
    content: TorrentContent,
    rel_path: PathBuf,
}

struct ItemGroup {
    index: usize,
    name: String,
    files: Vec<ItemFile>,
}

fn split_torrent_items(files: Vec<TorrentContent>) -> Vec<ItemGroup> {
    let mut by_dir: BTreeMap<String, Vec<ItemFile>> = BTreeMap::new();
    let mut root_files: Vec<ItemFile> = vec![];

    for file in files {
        let full_path = PathBuf::from(&file.name);
        let mut components = full_path.components();
        let Some(first) = components.next() else {
            continue;
        };
        match first {
            Component::Normal(dir) => {
                let rest = components.as_path();
                if rest.as_os_str().is_empty() {
                    root_files.push(ItemFile {
                        content: file,
                        rel_path: full_path,
                    });
                } else {
                    let dir_name = dir.to_string_lossy().to_string();
                    by_dir
                        .entry(dir_name)
                        .or_default()
                        .push(ItemFile {
                            content: file,
                            rel_path: rest.to_path_buf(),
                        });
                }
            }
            _ => root_files.push(ItemFile {
                content: file,
                rel_path: full_path,
            }),
        }
    }

    if root_files.is_empty() && by_dir.len() > 1 {
        by_dir
            .into_iter()
            .enumerate()
            .map(|(index, (name, files))| ItemGroup { index, name, files })
            .collect()
    } else {
        let mut all_files = root_files;
        for (_name, files) in by_dir {
            all_files.extend(files);
        }
        vec![ItemGroup {
            index: 0,
            name: String::new(),
            files: all_files,
        }]
    }
}

fn series_entry_for_index(
    meta: &TorrentMeta,
    item_count: usize,
    index: usize,
) -> Option<SeriesEntry> {
    let series = meta
        .series
        .iter()
        .find(|s| !s.entries.0.is_empty())
        .or(meta.series.first())?;
    if series.entries.0.len() == item_count {
        return series.entries.0.get(index).cloned();
    }
    if series.entries.0.len() == 1 {
        match series.entries.0[0] {
            SeriesEntry::Range(start, end) => {
                let entry = start + index as f32;
                if entry <= end {
                    return Some(SeriesEntry::Num(entry));
                }
            }
            SeriesEntry::Num(num) if item_count == 1 => return Some(SeriesEntry::Num(num)),
            SeriesEntry::Part(entry, part) if item_count == 1 => {
                return Some(SeriesEntry::Part(entry, part));
            }
            _ => {}
        }
    }
    if item_count > 1 {
        return Some(SeriesEntry::Num((index + 1) as f32));
    }
    None
}

pub(crate) fn library_item_meta(
    base: &TorrentMeta,
    item_title: String,
    series_entry: Option<SeriesEntry>,
) -> TorrentMeta {
    let mut meta = base.clone();
    meta.title = item_title;
    if let Some(entry) = series_entry {
        if let Some(series) = meta
            .series
            .iter_mut()
            .find(|s| !s.entries.0.is_empty())
            .or(meta.series.first_mut())
        {
            series.entries = SeriesEntries::new(vec![entry]);
        }
    }
    meta
}

fn library_file_path(path: &Path) -> PathBuf {
    let mut path_components = path.components();
    let file_name = path_components.next_back().unwrap();
    let dir_name = path_components.next_back().and_then(|dir_name| {
        if let Component::Normal(dir_name) = dir_name {
            let dir_name = dir_name.to_string_lossy().to_string();
            if let Some(disc) = DISK_PATTERN.captures(&dir_name).and_then(|c| c.get(1)) {
                return Some(format!("Disc {}", disc.as_str()));
            }
        }
        None
    });
    if let Some(dir_name) = dir_name {
        PathBuf::from(dir_name).join(file_name)
    } else {
        PathBuf::from(file_name)
    }
}

#[instrument(skip_all)]
pub async fn link_torrents_to_library(
    config: Arc<Config>,
    db: Arc<Database<'_>>,
    qbit: (&QbitConfig, &qbit::Api),
    mam: Arc<MaM<'_>>,
) -> Result<()> {
    let torrents = qbit
        .1
        .torrents(Some(TorrentListParams::default()))
        .await
        .context("qbit main data")?;

    for torrent in torrents {
        if torrent.progress < 1.0 {
            continue;
        }
        let library = find_library(&config, &torrent);
        let r = db.r_transaction()?;
        let mut existing_torrent: Option<Torrent> = r.get().primary(torrent.hash.clone())?;
        if let Some(t) = &mut existing_torrent {
            let library_name = library.and_then(|l| l.tag_filters().name.as_ref());
            if t.linker.as_ref() != library_name {
                let (_guard, rw) = db.rw_async().await?;
                t.linker = library_name.map(ToOwned::to_owned);
                rw.upsert(t.clone())?;
                rw.commit()?;
            }
            let category = if torrent.category.is_empty() {
                None
            } else {
                Some(torrent.category.as_str())
            };
            if t.category.as_deref() != category {
                let (_guard, rw) = db.rw_async().await?;
                t.category = category.map(ToOwned::to_owned);
                rw.upsert(t.clone())?;
                rw.commit()?;
            }
            if t.client_status.is_none() {
                let trackers = qbit.1.trackers(&torrent.hash).await?;
                if let Some(mam_tracker) = trackers.last()
                    && mam_tracker.msg == "torrent not registered with this tracker"
                {
                    {
                        let (_guard, rw) = db.rw_async().await?;
                        t.client_status = Some(ClientStatus::RemovedFromMam);
                        rw.upsert(t.clone())?;
                        rw.commit()?;
                    }
                    write_event(
                        &db,
                        Event::new(
                            Some(torrent.hash.clone()),
                            Some(t.mam_id),
                            EventType::RemovedFromMam,
                        ),
                    )
                    .await;
                }
            }
            if let Some(library_path) = &t.library_path {
                let Some(library) = find_library(&config, &torrent) else {
                    if t.library_mismatch != Some(LibraryMismatch::NoLibrary) {
                        debug!("no library: {library_path:?}",);
                        t.library_mismatch = Some(LibraryMismatch::NoLibrary);
                        let (_guard, rw) = db.rw_async().await?;
                        rw.upsert(t.clone())?;
                        rw.commit()?;
                    }
                    continue;
                };
                if !library_path.starts_with(library.library_dir()) {
                    let wanted = Some(LibraryMismatch::NewLibraryDir(
                        library.library_dir().clone(),
                    ));
                    if t.library_mismatch != wanted {
                        debug!(
                            "library differs: {library_path:?} != {:?}",
                            library.library_dir()
                        );
                        t.library_mismatch = wanted;
                        let (_guard, rw) = db.rw_async().await?;
                        rw.upsert(t.clone())?;
                        rw.commit()?;
                    }
                } else {
                    let dir = library_dir(config.exclude_narrator_in_library_dir, library, &t.meta);
                    let mut is_wrong = Some(library_path) != dir.as_ref();
                    let wanted = match dir {
                        Some(dir) => Some(LibraryMismatch::NewPath(dir)),
                        None => Some(LibraryMismatch::NoLibrary),
                    };

                    if t.library_mismatch != wanted {
                        if is_wrong {
                            // Try another attempt at matching with exclude_narrator flipped
                            let dir_2 = library_dir(
                                !config.exclude_narrator_in_library_dir,
                                library,
                                &t.meta,
                            );
                            if Some(library_path) == dir_2.as_ref() {
                                is_wrong = false
                            }
                        }
                        if is_wrong {
                            debug!("path differs: {library_path:?} != {:?}", wanted);
                            t.library_mismatch = wanted;
                            let (_guard, rw) = db.rw_async().await?;
                            rw.upsert(t.clone())?;
                            rw.commit()?;
                        } else if t.library_mismatch.is_some() {
                            t.library_mismatch = None;
                            let (_guard, rw) = db.rw_async().await?;
                            rw.upsert(t.clone())?;
                            rw.commit()?;
                        }
                    }
                }
                continue;
            }
            let library_items = r
                .scan()
                .secondary::<LibraryItem>(LibraryItemKey::torrent_id)?
                .range(torrent.hash.as_str()..=torrent.hash.as_str())?;
            if library_items
                .into_iter()
                .any(|item| item.ok().is_some_and(|item| item.library_path.is_some()))
            {
                continue;
            }
            if t.replaced_with.is_some() {
                continue;
            }
        }

        {
            let selected_torrent: Option<SelectedTorrent> = r.get().secondary::<SelectedTorrent>(
                SelectedTorrentKey::hash,
                Some(torrent.hash.clone()),
            )?;
            if let Some(selected_torrent) = selected_torrent {
                debug!(
                    "Finished Downloading torrent {} {}",
                    selected_torrent.mam_id, selected_torrent.meta.title
                );
                let (_guard, rw) = db.rw_async().await?;
                rw.remove(selected_torrent)?;
                rw.commit()?;
            }
        }
        let Some(library) = library else {
            trace!(
                "Could not find matching library for torrent \"{}\", save_path {}",
                torrent.name, torrent.save_path
            );
            continue;
        };

        if library.method() == LibraryLinkMethod::NoLink && existing_torrent.is_some() {
            continue;
        }

        let result = match_torrent(
            config.clone(),
            db.clone(),
            qbit,
            mam.clone(),
            &torrent.hash,
            &torrent,
            library,
            existing_torrent,
        )
        .await
        .context("match_torrent");
        update_errored_torrent(
            &db,
            ErroredTorrentId::Linker(torrent.hash.clone()),
            torrent.name,
            result,
        )
        .await;
    }

    Ok(())
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
async fn match_torrent(
    config: Arc<Config>,
    db: Arc<Database<'_>>,
    qbit: (&QbitConfig, &qbit::Api),
    mam: Arc<MaM<'_>>,
    hash: &str,
    torrent: &QbitTorrent,
    library: &Library,
    existing_torrent: Option<Torrent>,
) -> Result<()> {
    let files = qbit.1.files(hash, None).await?;
    let Some(mam_torrent) = mam.get_torrent_info(hash).await.context("get_mam_info")? else {
        bail!("Could not find torrent on mam");
    };
    let meta = match mam_torrent.as_meta() {
        Ok(meta) => meta,
        Err(err) => {
            if let MetaError::UnknownMediaType(_) = err {
                if let Some(on_invalid_torrent) = &qbit.0.on_invalid_torrent {
                    let qbit = qbit::Api::new_login_username_password(
                        &qbit.0.url,
                        &qbit.0.username,
                        &qbit.0.password,
                    )
                    .await?;

                    if let Some(category) = &on_invalid_torrent.category {
                        qbit.set_category(Some(vec![&torrent.hash]), category)
                            .await?;
                    }

                    if !on_invalid_torrent.tags.is_empty() {
                        qbit.add_tags(
                            Some(vec![&torrent.hash]),
                            on_invalid_torrent.tags.iter().map(Deref::deref).collect(),
                        )
                        .await?;
                    }
                }
                trace!("qbit updated");
            }
            return Err(err).context("as_meta");
        }
    };

    link_torrent(
        &config,
        qbit.0,
        &db,
        hash,
        torrent,
        files,
        library,
        mam_torrent,
        existing_torrent.as_ref(),
        &meta,
    )
    .await
    .context("link_torrent")
    .map_err(|err| anyhow::Error::new(TorrentMetaError(meta, err)))
}

#[instrument(skip_all)]
pub async fn refresh_metadata(
    config: &Config,
    db: &Database<'_>,
    mam: &MaM<'_>,
    id: String,
) -> Result<(Torrent, MaMTorrent)> {
    let Some(mut torrent): Option<Torrent> = db.r_transaction()?.get().primary(id)? else {
        bail!("Could not find torrent id");
    };
    debug!("refreshing metadata for torrent {}", torrent.meta.mam_id);
    let Some(mam_torrent) = mam
        .get_torrent_info_by_id(torrent.mam_id)
        .await
        .context("get_mam_info")?
    else {
        bail!("Could not find torrent \"{}\" on mam", torrent.meta.title);
    };
    let meta = mam_torrent.as_meta().context("as_meta")?;

    if torrent.meta != meta {
        update_torrent_meta(
            config,
            db,
            db.rw_async().await?,
            &mam_torrent,
            torrent.clone(),
            meta.clone(),
            true,
            false,
        )
        .await?;
        torrent.meta = meta;
    }
    Ok((torrent, mam_torrent))
}

#[instrument(skip_all)]
pub async fn refresh_metadata_relink(
    config: &Config,
    db: &Database<'_>,
    mam: &MaM<'_>,
    hash: String,
) -> Result<()> {
    let mut torrent = None;
    for qbit_conf in &config.qbittorrent {
        let qbit = match qbit::Api::new_login_username_password(
            &qbit_conf.url,
            &qbit_conf.username,
            &qbit_conf.password,
        )
        .await
        {
            Ok(qbit) => qbit,
            Err(err) => {
                error!("Error logging in to qbit {}: {err}", qbit_conf.url);
                continue;
            }
        };
        let mut torrents = match qbit
            .torrents(Some(TorrentListParams {
                hashes: Some(vec![hash.clone()]),
                ..TorrentListParams::default()
            }))
            .await
        {
            Ok(torrents) => torrents,
            Err(err) => {
                error!("Error getting torrents from qbit {}: {err}", qbit_conf.url);
                continue;
            }
        };
        let Some(t) = torrents.pop() else {
            continue;
        };
        torrent.replace((qbit_conf, qbit, t));
        break;
    }
    let Some((qbit_conf, qbit, qbit_torrent)) = torrent else {
        bail!("Could not find torrent in qbit");
    };
    let Some(library) = find_library(config, &qbit_torrent) else {
        bail!("Could not find matching library for torrent");
    };
    let files = qbit.files(&hash, None).await?;
    let (torrent, mam_torrent) = refresh_metadata(config, db, mam, hash.clone()).await?;
    let has_items = db
        .r_transaction()?
        .scan()
        .secondary::<LibraryItem>(LibraryItemKey::torrent_id)?
        .range(hash.as_str()..=hash.as_str())?
        .into_iter()
        .any(|item| item.ok().is_some());
    let library_path_changed = !has_items
        && torrent.library_path
            != library_dir(
                config.exclude_narrator_in_library_dir,
                library,
                &torrent.meta,
            );
    remove_library_files(config, db, &torrent, library_path_changed).await?;
    link_torrent(
        config,
        qbit_conf,
        db,
        &hash,
        &qbit_torrent,
        files,
        library,
        mam_torrent,
        Some(&torrent),
        &torrent.meta,
    )
    .await
    .context("link_torrent")
    .map_err(|err| anyhow::Error::new(TorrentMetaError(torrent.meta, err)))
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
async fn link_torrent(
    config: &Config,
    qbit_config: &QbitConfig,
    db: &Database<'_>,
    hash: &str,
    torrent: &QbitTorrent,
    files: Vec<TorrentContent>,
    library: &Library,
    mam_torrent: MaMTorrent,
    existing_torrent: Option<&Torrent>,
    meta: &TorrentMeta,
) -> Result<()> {
    let item_groups = split_torrent_items(files);
    let multi_item = item_groups.len() > 1;
    let mut library_items: Vec<LibraryItem> = vec![];
    let mut torrent_library_files = vec![];
    let mut torrent_selected_audio_format = None;
    let mut torrent_selected_ebook_format = None;
    let mut torrent_library_path = None;

    let mut existing_item_abs: BTreeMap<u32, String> = BTreeMap::new();
    if multi_item {
        let r = db.r_transaction()?;
        let items = r
            .scan()
            .secondary::<LibraryItem>(LibraryItemKey::torrent_id)?
            .range(hash..=hash)?;
        for item in items {
            if let Ok(item) = item
                && let Some(abs_id) = item.abs_id
            {
                existing_item_abs.insert(item.item_index, abs_id);
            }
        }
    }
    if library.tag_filters().method != LibraryLinkMethod::NoLink {
        let mut had_files = false;
        let item_count = item_groups.len();
        for group in item_groups {
            let item_files: Vec<TorrentContent> =
                group.files.iter().map(|f| f.content.clone()).collect();
            let selected_audio_format = select_format(
                &library.tag_filters().audio_types,
                &config.audio_types,
                &item_files,
            );
            let selected_ebook_format = select_format(
                &library.tag_filters().ebook_types,
                &config.ebook_types,
                &item_files,
            );
            if selected_audio_format.is_none() && selected_ebook_format.is_none() {
                debug!("Skipping item without wanted formats: {}", group.name);
                continue;
            }

            let item_title = if multi_item {
                group.name.clone()
            } else {
                meta.title.clone()
            };
            let series_entry =
                if multi_item { series_entry_for_index(meta, item_count, group.index) } else { None };
            let item_meta = library_item_meta(meta, item_title.clone(), series_entry.clone());
            let Some(mut dir) = library_dir(
                config.exclude_narrator_in_library_dir,
                library,
                &item_meta,
            ) else {
                bail!("Torrent has no author");
            };
            if config.exclude_narrator_in_library_dir
                && !item_meta.narrators.is_empty()
                && dir.exists()
            {
                dir = library_dir(false, library, &item_meta).unwrap();
            }
            let metadata = abs::create_metadata(&mam_torrent, &item_meta);

            let mut library_files = vec![];
            create_dir_all(&dir).await?;
            for file in group.files {
                let span = span!(Level::TRACE, "file: {:?}", file.content.name);
                let _s = span.enter();
                let file_name = &file.content.name;
                if !(selected_audio_format
                    .as_ref()
                    .is_some_and(|ext| file_name.ends_with(ext))
                    || selected_ebook_format
                        .as_ref()
                        .is_some_and(|ext| file_name.ends_with(ext)))
                {
                    debug!("Skiping \"{}\"", file_name);
                    continue;
                }
                let file_path = library_file_path(&file.rel_path);
                if let Some(sub_dir) = file_path.parent() {
                    create_dir_all(dir.join(sub_dir)).await?;
                }
                let library_path = dir.join(&file_path);
                library_files.push(file_path.clone());
                let download_path =
                    map_path(&qbit_config.path_mapping, &torrent.save_path).join(file_name);
                match library.method() {
                    LibraryLinkMethod::Hardlink => {
                        hard_link(&download_path, &library_path, &file_path)?
                    }
                    LibraryLinkMethod::HardlinkOrCopy => {
                        hard_link(&download_path, &library_path, &file_path)
                            .or_else(|_| copy(&download_path, &library_path))?
                    }
                    LibraryLinkMethod::Copy => copy(&download_path, &library_path)?,
                    LibraryLinkMethod::HardlinkOrSymlink => {
                        hard_link(&download_path, &library_path, &file_path)
                            .or_else(|_| symlink(&download_path, &library_path))?
                    }
                    LibraryLinkMethod::Symlink => symlink(&download_path, &library_path)?,
                    LibraryLinkMethod::NoLink => {}
                };
            }
            library_files.sort();

            let file = File::create(dir.join("metadata.json"))?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer(&mut writer, &metadata)?;
            writer.flush()?;

            had_files = true;
            if !multi_item {
                torrent_library_path = Some(dir.clone());
                torrent_library_files = library_files.clone();
                torrent_selected_audio_format = selected_audio_format.clone();
                torrent_selected_ebook_format = selected_ebook_format.clone();
            }

            let item_id = format!("{hash}:{}", group.index);
            library_items.push(LibraryItem {
                id: item_id,
                torrent_id: hash.to_owned(),
                item_index: group.index as u32,
                item_name: group.name.clone(),
                item_title,
                series_entry,
                library_path: Some(dir),
                library_files,
                selected_audio_format,
                selected_ebook_format,
                title_search: normalize_title(&item_meta.title),
                abs_id: if multi_item {
                    existing_item_abs.get(&(group.index as u32)).cloned()
                } else {
                    existing_torrent.and_then(|t| t.abs_id.clone())
                },
                created_at: existing_torrent
                    .map(|t| t.created_at)
                    .unwrap_or_else(Timestamp::now),
            });
        }

        if !had_files {
            bail!("Could not find any wanted formats in torrent");
        }
    }

    {
        let (_guard, rw) = db.rw_async().await?;
        if multi_item {
            let existing_items = rw
                .scan()
                .secondary::<LibraryItem>(LibraryItemKey::torrent_id)?
                .range(hash..=hash)?;
            for item in existing_items {
                if let Ok(item) = item {
                    rw.remove(item)?;
                }
            }
        }
        for item in &library_items {
            rw.upsert(item.clone())?;
        }
        rw.upsert(Torrent {
            id: hash.to_owned(),
            id_is_hash: true,
            mam_id: meta.mam_id,
            abs_id: if multi_item {
                None
            } else {
                existing_torrent.and_then(|t| t.abs_id.clone())
            },
            goodreads_id: existing_torrent.and_then(|t| t.goodreads_id),
            library_path: torrent_library_path.clone(),
            library_files: torrent_library_files,
            linker: library.tag_filters().name.clone(),
            category: if torrent.category.is_empty() {
                None
            } else {
                Some(torrent.category.clone())
            },
            selected_audio_format: torrent_selected_audio_format,
            selected_ebook_format: torrent_selected_ebook_format,
            title_search: normalize_title(&meta.title),
            meta: meta.clone(),
            created_at: existing_torrent
                .map(|t| t.created_at)
                .unwrap_or_else(Timestamp::now),
            replaced_with: existing_torrent.and_then(|t| t.replaced_with.clone()),
            request_matadata_update: false,
            library_mismatch: None,
            client_status: existing_torrent.and_then(|t| t.client_status.clone()),
        })?;
        rw.commit()?;
    }

    if let Some(library_path) = torrent_library_path {
        write_event(
            db,
            Event::new(
                Some(hash.to_owned()),
                Some(meta.mam_id),
                EventType::Linked {
                    linker: library.tag_filters().name.clone(),
                    library_path,
                },
            ),
        )
        .await;
    }

    Ok(())
}

pub fn map_path(path_mapping: &BTreeMap<PathBuf, PathBuf>, save_path: &str) -> PathBuf {
    let mut path = PathBuf::from(save_path);
    for (from, to) in path_mapping.iter().rev() {
        if path.starts_with(from) {
            let mut components = path.components();
            for _ in from {
                components.next();
            }
            path = to.join(components.as_path());
            break;
        }
    }
    path
}

pub fn find_library<'a>(config: &'a Config, torrent: &QbitTorrent) -> Option<&'a Library> {
    config
        .libraries
        .iter()
        .filter(|l| match l {
            Library::ByDir(l) => PathBuf::from(&torrent.save_path).starts_with(&l.download_dir),
            Library::ByCategory(l) => torrent.category == l.category,
        })
        .find(|l| {
            let filters = l.tag_filters();
            if filters
                .deny_tags
                .iter()
                .any(|tag| torrent.tags.split(", ").any(|t| t == tag.as_str()))
            {
                return false;
            }
            if filters.allow_tags.is_empty() {
                return true;
            }
            filters
                .allow_tags
                .iter()
                .any(|tag| torrent.tags.split(", ").any(|t| t == tag.as_str()))
        })
}

pub fn library_dir(
    exclude_narrator_in_library_dir: bool,
    library: &Library,
    meta: &TorrentMeta,
) -> Option<PathBuf> {
    let author = meta.authors.first()?;
    let mut dir = match meta
        .series
        .iter()
        .find(|s| !s.entries.0.is_empty())
        .or(meta.series.first())
    {
        Some(series) => PathBuf::from(sanitize_filename::sanitize(author).to_string())
            .join(sanitize_filename::sanitize(&series.name).to_string())
            .join(
                sanitize_filename::sanitize(if series.entries.0.is_empty() {
                    meta.title.clone()
                } else {
                    format!("{} #{} - {}", series.name, series.entries, meta.title)
                })
                .to_string(),
            ),
        None => PathBuf::from(sanitize_filename::sanitize(author).to_string())
            .join(sanitize_filename::sanitize(&meta.title).to_string()),
    };
    if let Some((edition, _)) = &meta.edition {
        dir.set_file_name(
            sanitize_filename::sanitize(format!(
                "{}, {}",
                dir.file_name().unwrap().to_string_lossy(),
                edition
            ))
            .to_string(),
        );
    }
    if let Some(narrator) = meta.narrators.first()
        && !exclude_narrator_in_library_dir
    {
        dir.set_file_name(
            sanitize_filename::sanitize(format!(
                "{} {{{}}}",
                dir.file_name().unwrap().to_string_lossy(),
                narrator
            ))
            .to_string(),
        );
    }
    let dir = library.library_dir().join(dir);
    Some(dir)
}

fn select_format(
    overridden_wanted_formats: &Option<Vec<String>>,
    wanted_formats: &[String],
    files: &[TorrentContent],
) -> Option<String> {
    overridden_wanted_formats
        .as_deref()
        .unwrap_or(wanted_formats)
        .iter()
        .map(|ext| {
            let ext = ext.to_lowercase();
            if ext.starts_with(".") {
                ext.clone()
            } else {
                format!(".{ext}")
            }
        })
        .find(|ext| files.iter().any(|f| f.name.to_lowercase().ends_with(ext)))
}

#[instrument(skip_all)]
fn hard_link(download_path: &Path, library_path: &Path, file_path: &Path) -> Result<()> {
    debug!("linking: {:?} -> {:?}", download_path, library_path);
    fs::hard_link(download_path, library_path).or_else(|err| {
            if err.kind() == ErrorKind::AlreadyExists {
                trace!("AlreadyExists: {}", err);
                let download_id = get_file_id(download_path);
                trace!("got 1: {download_id:?}");
                let library_id = get_file_id(library_path);
                trace!("got 2: {library_id:?}");
                if let (Ok(download_id), Ok(library_id)) = (download_id, library_id) {
                    trace!("got both");
                    if download_id == library_id {
                        trace!("both match");
                        return Ok(());
                    } else {
                        trace!("no match");
                        bail!(
                            "File \"{:?}\" already exists, torrent file size: {}, library file size: {}",
                            file_path,
                            fs::metadata(download_path).map_or("?".to_string(), |s| Size::from_bytes(file_size(&s)).to_string()),
                            fs::metadata(library_path).map_or("?".to_string(), |s| Size::from_bytes(file_size(&s)).to_string())
                        );
                    }
                }
            }
            Err(err.into())
        })?;
    Ok(())
}

#[instrument(skip_all)]
fn copy(download_path: &Path, library_path: &Path) -> Result<()> {
    debug!("copying: {:?} -> {:?}", download_path, library_path);
    fs::copy(download_path, library_path)?;
    Ok(())
}

#[instrument(skip_all)]
fn symlink(download_path: &Path, library_path: &Path) -> Result<()> {
    debug!("symlinking: {:?} -> {:?}", download_path, library_path);
    #[cfg(target_family = "unix")]
    std::os::unix::fs::symlink(download_path, library_path)?;
    #[cfg(target_family = "windows")]
    bail!("symlink is not supported on Windows");
    #[allow(unreachable_code)]
    Ok(())
}

pub fn file_size(m: &Metadata) -> u64 {
    #[cfg(target_family = "unix")]
    return m.size();
    #[cfg(target_family = "windows")]
    return m.file_size();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_path() {
        let mut mappings = BTreeMap::new();
        mappings.insert(PathBuf::from("/downloads"), PathBuf::from("/books"));
        mappings.insert(
            PathBuf::from("/downloads/audiobooks"),
            PathBuf::from("/audiobooks"),
        );
        mappings.insert(PathBuf::from("/audiobooks"), PathBuf::from("/audiobooks"));

        assert_eq!(
            map_path(&mappings, "/downloads/torrent"),
            PathBuf::from("/books/torrent")
        );
        assert_eq!(
            map_path(&mappings, "/downloads/audiobooks/torrent"),
            PathBuf::from("/audiobooks/torrent")
        );
        assert_eq!(
            map_path(&mappings, "/downloads/audiobooks/torrent/deep"),
            PathBuf::from("/audiobooks/torrent/deep")
        );
        assert_eq!(
            map_path(&mappings, "/audiobooks/torrent"),
            PathBuf::from("/audiobooks/torrent")
        );
        assert_eq!(
            map_path(&mappings, "/ebooks/torrent"),
            PathBuf::from("/ebooks/torrent")
        );
    }
}
