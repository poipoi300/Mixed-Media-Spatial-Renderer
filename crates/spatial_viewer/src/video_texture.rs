//! Playing videos draw from one GPU texture each, which every decoded frame
//! is written into in place.
//!
//! A frame is never a new [`Image`] asset. An asset would go through Bevy's
//! per-frame upload budget, and the material pointed at it would be rebuilt.
//! Bevy drops the previous prepared material as soon as the change is
//! extracted, so a frame the budget deferred left the billboard with no
//! material, and it was not drawn for a frame (a black flicker).
//!
//! Instead the texture lives only in the render world, under a handle
//! reserved in the main world that has no asset behind it. The render world
//! creates the texture from the first frame, before materials are prepared,
//! so the material can be pointed at it in the same frame. Later frames
//! overwrite it in place without touching the material. Only the billboard's
//! surface and material hold the texture, so it is freed with them (through
//! Bevy's usual unused-asset path) once the billboard no longer draws it.

use std::collections::HashMap;

use bevy::{
    pbr::PreparedMaterial,
    prelude::*,
    render::{
        render_asset::{prepare_assets, RenderAssets},
        render_resource::{
            Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, TextureAspect, TextureDataOrder,
            TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
            TextureViewDescriptor,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::{DefaultImageSampler, GpuImage},
        MainWorld, Render, RenderApp, RenderSet,
    },
};

use crate::media_decode::DecodedImage;

const VIDEO_FRAME_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

/// Which texture a video billboard's frames are written into, on the
/// billboard's entity. It does not keep the texture alive: the billboard's
/// surface does, for as long as it draws it. A texture has one size for its
/// whole life; frames of another size get a new one.
#[derive(Component, Clone, Debug)]
pub struct VideoFrameTexture {
    id: AssetId<Image>,
    side: u32,
}

impl VideoFrameTexture {
    /// Reserves a texture for frames of `side`, with the strong handle that
    /// keeps it alive.
    pub fn reserve(images: &Assets<Image>, side: u32) -> (Self, Handle<Image>) {
        let handle = images.reserve_handle();
        let texture = Self {
            id: handle.id(),
            side,
        };
        (texture, handle)
    }

    /// Whether `surface_texture` is this texture, and it takes frames of
    /// `side`.
    pub fn is(&self, surface_texture: &Handle<Image>, side: u32) -> bool {
        surface_texture.id() == self.id && side == self.side
    }
}

/// Frames waiting to be written into their texture, at most one per
/// texture: a newer frame replaces one that was never written.
#[derive(Resource, Default)]
pub struct VideoFrameUploads {
    frames: HashMap<AssetId<Image>, VideoFrameUpload>,
}

impl VideoFrameUploads {
    /// Queues `frame` to be written into `texture`, which must have been
    /// reserved for frames of its size (see [`VideoFrameTexture::is`]).
    pub fn queue(&mut self, texture: &Handle<Image>, frame: DecodedImage) {
        let side = frame.side as usize;
        assert_eq!(
            frame.rgba.len(),
            side * side * 4,
            "a video frame must be a whole RGBA square"
        );
        self.frames.insert(
            texture.id(),
            VideoFrameUpload {
                // Kept strong until written, so the texture cannot be freed
                // between queueing and writing.
                handle: texture.clone(),
                side: frame.side,
                rgba: frame.rgba,
            },
        );
    }
}

struct VideoFrameUpload {
    handle: Handle<Image>,
    side: u32,
    rgba: Vec<u8>,
}

/// The frames extracted this frame, written before materials are prepared.
#[derive(Resource, Default)]
struct ExtractedVideoFrames {
    frames: Vec<VideoFrameUpload>,
}

pub struct VideoTexturePlugin;

impl Plugin for VideoTexturePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VideoFrameUploads>();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ExtractedVideoFrames>()
            .add_systems(ExtractSchedule, extract_video_frames)
            .add_systems(
                Render,
                write_video_frames
                    .in_set(RenderSet::PrepareAssets)
                    .after(prepare_assets::<GpuImage>)
                    .before(prepare_assets::<PreparedMaterial<StandardMaterial>>),
            );
    }
}

fn extract_video_frames(
    mut main_world: ResMut<MainWorld>,
    mut extracted: ResMut<ExtractedVideoFrames>,
) {
    let mut uploads = main_world.resource_mut::<VideoFrameUploads>();
    extracted
        .frames
        .extend(uploads.frames.drain().map(|(_, upload)| upload));
}

fn write_video_frames(
    mut extracted: ResMut<ExtractedVideoFrames>,
    mut gpu_images: ResMut<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    default_sampler: Res<DefaultImageSampler>,
) {
    for upload in extracted.frames.drain(..) {
        let extent = Extent3d {
            width: upload.side,
            height: upload.side,
            depth_or_array_layers: 1,
        };
        let id = upload.handle.id();
        if let Some(gpu_image) = gpu_images.get(id) {
            render_queue.write_texture(
                ImageCopyTexture {
                    texture: &gpu_image.texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &upload.rgba,
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.side * 4),
                    rows_per_image: Some(upload.side),
                },
                extent,
            );
            continue;
        }
        let texture = render_device.create_texture_with_data(
            &render_queue,
            &TextureDescriptor {
                label: Some("video_frame"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: VIDEO_FRAME_FORMAT,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::default(),
            &upload.rgba,
        );
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        gpu_images.insert(
            id,
            GpuImage {
                texture,
                texture_view,
                texture_format: VIDEO_FRAME_FORMAT,
                sampler: (**default_sampler).clone(),
                size: UVec2::splat(upload.side),
                mip_level_count: 1,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(side: u32, value: u8) -> DecodedImage {
        DecodedImage::full(side, vec![value; (side * side * 4) as usize])
    }

    #[test]
    fn a_reserved_texture_is_its_own_handle_at_its_own_size() {
        let images = Assets::<Image>::default();
        let (texture, handle) = VideoFrameTexture::reserve(&images, 2);
        let (_, other_handle) = VideoFrameTexture::reserve(&images, 2);

        assert!(texture.is(&handle, 2));
        assert!(!texture.is(&handle, 4));
        assert!(!texture.is(&other_handle, 2));
    }

    #[test]
    fn a_newer_frame_replaces_one_never_written() {
        let images = Assets::<Image>::default();
        let (_, texture) = VideoFrameTexture::reserve(&images, 2);
        let (_, other) = VideoFrameTexture::reserve(&images, 2);
        let mut uploads = VideoFrameUploads::default();

        uploads.queue(&texture, frame(2, 1));
        uploads.queue(&other, frame(2, 2));
        uploads.queue(&texture, frame(2, 3));

        assert_eq!(uploads.frames.len(), 2);
        assert_eq!(uploads.frames[&texture.id()].rgba[0], 3);
        assert_eq!(uploads.frames[&other.id()].rgba[0], 2);
    }

    #[test]
    #[should_panic(expected = "a video frame must be a whole RGBA square")]
    fn a_frame_short_of_its_side_is_refused() {
        let images = Assets::<Image>::default();
        let (_, texture) = VideoFrameTexture::reserve(&images, 2);
        VideoFrameUploads::default().queue(&texture, DecodedImage::full(2, vec![0; 8]));
    }
}
