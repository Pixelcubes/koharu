use crate::types::{TextDirection, TextRegion};
use image::{
    DynamicImage, GrayImage, Luma, Rgb, RgbImage,
    imageops::{self},
};
use imageproc::{
    distance_transform::Norm,
    geometric_transformations::{Interpolation, Projection, warp_into},
    morphology::{close, dilate},
};

const FINAL_MASK_DILATE_RADIUS: u8 = 2;
// Bridges small gaps left by the segmentation net under-predicting glyph
// pixels over busy/textured backgrounds (see `refine_segmentation_mask`).
// Closing (dilate-then-erode) only fills holes near existing mask pixels
// and is "extensive" (it never removes an already-detected pixel), so
// unlike plain dilation it can't turn this into a broad region-fill (that
// approach was tried for AOT/Lama and reverted, see 8e92960f).
//
// The radius scales with each block's own detected font size instead of
// being a fixed pixel count: manga pages vary enormously in resolution
// and lettering size, so a fixed radius would be negligible on a large
// high-res scan and too aggressive on a small/low-res one.
const HOLE_CLOSE_FONT_RATIO: f32 = 0.08;
const HOLE_CLOSE_MIN_RADIUS: u8 = 1;
const HOLE_CLOSE_MAX_RADIUS: u8 = 6;

pub type Quad = [[f32; 2]; 4];

#[derive(Debug, Clone)]
pub struct ComicTextDetection {
    pub shrink_map: GrayImage,
    pub threshold_map: GrayImage,
    pub line_polygons: Vec<Quad>,
    pub text_blocks: Vec<TextRegion>,
    pub mask: GrayImage,
}

pub fn refine_segmentation_mask(
    _image: &DynamicImage,
    pred_mask: &GrayImage,
    blocks: &[TextRegion],
) -> GrayImage {
    let width = pred_mask.width();
    let height = pred_mask.height();

    if blocks.is_empty() {
        return GrayImage::new(width, height);
    }

    // Extract expanded bounding boxes globally to validate intersection constraints.
    let expanded_bounds: Vec<[u32; 4]> = blocks
        .iter()
        .map(|b| expanded_text_block_crop_bounds(width, height, b))
        .collect();

    // Rasterize the union of expanded text block bounds once to avoid an
    // O(width * height * blocks) per-pixel rectangle membership test.
    let mut in_bounds_mask = GrayImage::new(width, height);
    for &[x1, y1, x2, y2] in &expanded_bounds {
        for y in y1..y2 {
            for x in x1..x2 {
                in_bounds_mask.put_pixel(x, y, Luma([255]));
            }
        }
    }

    // Apply a threshold mask: Pixels are preserved exclusively if their probability
    // exceeds the core threshold (`super::BINARY_THRESHOLD`) and they reside within a known TextRegion geometry.
    let base = GrayImage::from_fn(width, height, |x, y| {
        if in_bounds_mask.get_pixel(x, y)[0] != 0
            && pred_mask.get_pixel(x, y)[0] > super::BINARY_THRESHOLD
        {
            Luma([255])
        } else {
            Luma([0])
        }
    });

    // Close gaps per block, scaled to that block's own font size, rather
    // than one fixed radius for the whole page — a page can mix tiny SFX
    // lettering with large splash-panel text.
    let mut closed = base.clone();
    for (block, &bounds) in blocks.iter().zip(&expanded_bounds) {
        let radius = hole_close_radius(block);
        let padded = expand_rect(bounds, width, height, u32::from(radius));
        if count_nonzero_in_rect(&base, padded) == 0 {
            continue;
        }
        let crop = imageops::crop_imm(
            &base,
            padded[0],
            padded[1],
            padded[2] - padded[0],
            padded[3] - padded[1],
        )
        .to_image();
        let local_closed = close(&crop, Norm::LInf, radius);
        merge_local_closed(&mut closed, &local_closed, padded, bounds);
    }

    let dilated = dilate(&closed, Norm::L1, FINAL_MASK_DILATE_RADIUS);

    // Final clipping pass: Ensure the dilated mask never escapes the block boundaries
    // even if it thickens beyond its original source pixel edges.
    GrayImage::from_fn(width, height, |x, y| {
        if in_bounds_mask.get_pixel(x, y)[0] != 0 {
            *dilated.get_pixel(x, y)
        } else {
            Luma([0])
        }
    })
}

fn hole_close_radius(block: &TextRegion) -> u8 {
    let font = block
        .detected_font_size_px
        .unwrap_or_else(|| block.width.min(block.height).max(1.0));
    ((font * HOLE_CLOSE_FONT_RATIO).round() as u8)
        .clamp(HOLE_CLOSE_MIN_RADIUS, HOLE_CLOSE_MAX_RADIUS)
}

fn expand_rect([x1, y1, x2, y2]: [u32; 4], width: u32, height: u32, pad: u32) -> [u32; 4] {
    [
        x1.saturating_sub(pad),
        y1.saturating_sub(pad),
        x2.saturating_add(pad).min(width),
        y2.saturating_add(pad).min(height),
    ]
}

fn count_nonzero_in_rect(mask: &GrayImage, [x1, y1, x2, y2]: [u32; 4]) -> u32 {
    let mut count = 0;
    for y in y1..y2 {
        for x in x1..x2 {
            if mask.get_pixel(x, y).0[0] > 0 {
                count += 1;
            }
        }
    }
    count
}

// Merges a locally-closed crop back into the page mask, restricted to the
// block's own (unpadded) bounds so one block's closing can't spill into a
// neighboring block's territory even when their padded working windows
// overlap. Only ever sets pixels (never clears), matching closing's
// extensive property — the original `base` detections underneath are
// never lost even at crop-window edges.
fn merge_local_closed(out: &mut GrayImage, local: &GrayImage, padded: [u32; 4], clip: [u32; 4]) {
    for local_y in 0..local.height() {
        let y = padded[1] + local_y;
        if y < clip[1] || y >= clip[3] {
            continue;
        }
        for local_x in 0..local.width() {
            if local.get_pixel(local_x, local_y).0[0] == 0 {
                continue;
            }
            let x = padded[0] + local_x;
            if x < clip[0] || x >= clip[2] {
                continue;
            }
            out.put_pixel(x, y, Luma([255]));
        }
    }
}

pub fn crop_text_block_bbox(image: &DynamicImage, block: &TextRegion) -> DynamicImage {
    let [x1, y1, x2, y2] = expanded_text_block_crop_bounds(image.width(), image.height(), block);
    image.crop_imm(x1, y1, x2.saturating_sub(x1), y2.saturating_sub(y1))
}

pub fn extract_text_block_regions(image: &DynamicImage, block: &TextRegion) -> Vec<DynamicImage> {
    let Some(line_polygons) = block.line_polygons.as_ref() else {
        return vec![crop_text_block_bbox(image, block)];
    };
    if line_polygons.is_empty() {
        return vec![crop_text_block_bbox(image, block)];
    }

    let rgb = image.to_rgb8();
    let mut regions = Vec::with_capacity(line_polygons.len());
    for line in line_polygons {
        if let Some(region) = warp_line_region(&rgb, block, line) {
            regions.push(DynamicImage::ImageRgb8(region));
        }
    }

    if regions.is_empty() {
        vec![crop_text_block_bbox(image, block)]
    } else {
        regions
    }
}

pub fn expanded_text_block_crop_bounds(
    image_width: u32,
    image_height: u32,
    block: &TextRegion,
) -> [u32; 4] {
    // Every detector gets the same small, font-scaled padding here, not
    // just "ctd" — this box is used as a hard clip on the erase mask (see
    // `refine_segmentation_mask`), both before and after gap-closing runs.
    // A zero-slack box silently re-excludes glyph pixels regardless of how
    // good the mask/closing logic is if the detector's box is off by even
    // a couple pixels, which is normal localization jitter for any
    // detector, not something specific to CTD's shrink-map under-reporting.
    let mut min_x = block.x;
    let mut min_y = block.y;
    let mut max_x = block.x + block.width;
    let mut max_y = block.y + block.height;

    if let Some(line_polygons) = block.line_polygons.as_ref() {
        for line in line_polygons {
            let quad = maybe_expand_ctd_line(block, line);
            let bbox = quad_bbox(&quad);
            min_x = min_x.min(bbox[0]);
            min_y = min_y.min(bbox[1]);
            max_x = max_x.max(bbox[2]);
            max_y = max_y.max(bbox[3]);
        }
    }

    let font = block
        .detected_font_size_px
        .unwrap_or_else(|| block.width.min(block.height).max(1.0));
    let base_pad = (font * 0.08).max(2.0);
    let (pad_x, pad_y) = match block.source_direction.unwrap_or(TextDirection::Horizontal) {
        TextDirection::Horizontal => ((font * 0.12).max(base_pad), (font * 0.18).max(base_pad)),
        TextDirection::Vertical => ((font * 0.18).max(base_pad), (font * 0.12).max(base_pad)),
    };

    let x1 = (min_x - pad_x)
        .floor()
        .clamp(0.0, image_width.saturating_sub(1) as f32) as u32;
    let y1 = (min_y - pad_y)
        .floor()
        .clamp(0.0, image_height.saturating_sub(1) as f32) as u32;
    let x2 = (max_x + pad_x)
        .ceil()
        .clamp(x1 as f32 + 1.0, image_width as f32) as u32;
    let y2 = (max_y + pad_y)
        .ceil()
        .clamp(y1 as f32 + 1.0, image_height as f32) as u32;
    [x1, y1, x2, y2]
}

fn warp_line_region(image: &RgbImage, block: &TextRegion, line: &Quad) -> Option<RgbImage> {
    let expanded = maybe_expand_ctd_line(block, line);
    let clipped = clip_quad(&expanded, image.width() as f32, image.height() as f32);
    let bbox = quad_bbox(&clipped);
    let x1 = bbox[0].floor().max(0.0) as u32;
    let y1 = bbox[1].floor().max(0.0) as u32;
    let x2 = bbox[2].ceil().min(image.width() as f32) as u32;
    let y2 = bbox[3].ceil().min(image.height() as f32) as u32;
    if x2 <= x1 || y2 <= y1 {
        return None;
    }

    let cropped = imageops::crop_imm(image, x1, y1, x2 - x1, y2 - y1).to_image();
    let mut src = clipped;
    for point in &mut src {
        point[0] -= x1 as f32;
        point[1] -= y1 as f32;
    }

    let (norm_v, norm_h) = quad_axis_lengths(&src);
    if norm_v <= 0.0 || norm_h <= 0.0 {
        return None;
    }

    let direction = block.source_direction.unwrap_or(TextDirection::Horizontal);
    let text_height = match direction {
        TextDirection::Horizontal => norm_v.max(1.0).round() as u32,
        TextDirection::Vertical => norm_h.max(1.0).round() as u32,
    }
    .max(1);
    let ratio = norm_v / norm_h;

    let (width, height, rotate_vertical) = match direction {
        TextDirection::Horizontal => {
            let h = text_height.max(1);
            let w = ((text_height as f32 / ratio).round() as u32).max(1);
            (w, h, false)
        }
        TextDirection::Vertical => {
            let w = text_height.max(1);
            let h = ((text_height as f32 * ratio).round() as u32).max(1);
            (w, h, true)
        }
    };

    let dst = [
        (0.0f32, 0.0f32),
        ((width.saturating_sub(1)) as f32, 0.0f32),
        (
            (width.saturating_sub(1)) as f32,
            (height.saturating_sub(1)) as f32,
        ),
        (0.0f32, (height.saturating_sub(1)) as f32),
    ];
    let src = quad_to_tuples(&src);
    let projection = Projection::from_control_points(src, dst)?;

    let mut region = RgbImage::from_pixel(width, height, Rgb([0, 0, 0]));
    warp_into(
        &cropped,
        projection,
        Interpolation::Bilinear,
        imageproc::geometric_transformations::Border::Constant(Rgb([0, 0, 0])),
        &mut region,
    );

    if rotate_vertical {
        Some(imageops::rotate270(&region))
    } else {
        Some(region)
    }
}

fn maybe_expand_ctd_line(block: &TextRegion, line: &Quad) -> Quad {
    let should_expand = block.detector.as_deref() == Some("ctd")
        && block.source_direction == Some(TextDirection::Horizontal);
    if !should_expand {
        return *line;
    }

    let expand_size = (block.detected_font_size_px.unwrap_or(0.0) * 0.1).max(3.0);
    let angle = block.rotation_deg.unwrap_or(0.0).to_radians();
    let sin = angle.sin();
    let cos = angle.cos();
    let signs = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

    let mut out = *line;
    for (index, point) in out.iter_mut().enumerate() {
        point[0] += signs[index][0] * sin * expand_size;
        point[1] += signs[index][1] * cos * expand_size;
    }
    out
}

fn clip_quad(quad: &Quad, width: f32, height: f32) -> Quad {
    let mut clipped = *quad;
    for point in &mut clipped {
        point[0] = point[0].clamp(0.0, width);
        point[1] = point[1].clamp(0.0, height);
    }
    clipped
}

fn quad_bbox(quad: &Quad) -> [f32; 4] {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for point in quad {
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }
    [min_x, min_y, max_x, max_y]
}

fn quad_to_tuples(quad: &Quad) -> [(f32, f32); 4] {
    [
        (quad[0][0], quad[0][1]),
        (quad[1][0], quad[1][1]),
        (quad[2][0], quad[2][1]),
        (quad[3][0], quad[3][1]),
    ]
}

fn quad_axis_lengths(quad: &Quad) -> (f32, f32) {
    let midpoints = [
        midpoint(quad[0], quad[1]),
        midpoint(quad[1], quad[2]),
        midpoint(quad[2], quad[3]),
        midpoint(quad[3], quad[0]),
    ];
    let vec_v = [
        midpoints[2][0] - midpoints[0][0],
        midpoints[2][1] - midpoints[0][1],
    ];
    let vec_h = [
        midpoints[1][0] - midpoints[3][0],
        midpoints[1][1] - midpoints[3][1],
    ];
    (vector_norm(vec_v), vector_norm(vec_h))
}

fn midpoint(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

fn vector_norm(vector: [f32; 2]) -> f32 {
    (vector[0] * vector[0] + vector[1] * vector[1]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refine_segmentation_mask_erases_when_blocks_are_missing() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, Rgb([255, 255, 255])));
        let pred_mask = GrayImage::from_fn(16, 16, |x, y| {
            if (4..12).contains(&x) && (5..11).contains(&y) {
                Luma([200])
            } else {
                Luma([0])
            }
        });

        let mask = refine_segmentation_mask(&image, &pred_mask, &[]);
        assert_eq!(mask.get_pixel(0, 0)[0], 0);
        assert_eq!(mask.get_pixel(8, 8)[0], 0); // No blocks, must be wiped cleanly
    }

    #[test]
    fn refine_segmentation_mask_closes_small_internal_gaps() {
        // Simulates a glyph stroke over a busy background where the
        // segmentation net under-predicted a small patch in the middle of
        // an otherwise well-detected region (below BINARY_THRESHOLD there,
        // at/above it everywhere around it).
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 40, Rgb([255, 255, 255])));
        let pred_mask = GrayImage::from_fn(40, 40, |x, y| {
            let in_glyph = (10..30).contains(&x) && (10..30).contains(&y);
            let in_gap = (18..20).contains(&x) && (18..20).contains(&y);
            if in_glyph && !in_gap {
                Luma([200])
            } else {
                Luma([0])
            }
        });

        let block = TextRegion {
            x: 8.0,
            y: 8.0,
            width: 24.0,
            height: 24.0,
            detected_font_size_px: Some(24.0),
            ..Default::default()
        };

        let mask = refine_segmentation_mask(&image, &pred_mask, &[block]);

        // The under-predicted gap in the middle of the glyph is bridged.
        assert_eq!(mask.get_pixel(18, 18)[0], 255);
        assert_eq!(mask.get_pixel(19, 19)[0], 255);
        // Closing must not have grown the mask far beyond the glyph's own
        // footprint into unrelated background outside the text block.
        assert_eq!(mask.get_pixel(2, 2)[0], 0);
        assert_eq!(mask.get_pixel(38, 38)[0], 0);
    }

    #[test]
    fn refine_segmentation_mask_close_radius_scales_with_font_size() {
        // A large-font block's gap is wide enough that a small, fixed
        // radius wouldn't bridge it; the font-scaled radius should still
        // close it (this is the case a flat pixel constant would miss on
        // a high-res page with large lettering).
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(60, 60, Rgb([255, 255, 255])));
        let gap_mask = GrayImage::from_fn(60, 60, |x, y| {
            let in_glyph = (10..40).contains(&x) && (10..40).contains(&y);
            let in_gap = (20..30).contains(&x) && (20..30).contains(&y);
            if in_glyph && !in_gap {
                Luma([200])
            } else {
                Luma([0])
            }
        });
        let large_font_block = TextRegion {
            x: 8.0,
            y: 8.0,
            width: 34.0,
            height: 34.0,
            detected_font_size_px: Some(150.0), // clamps to HOLE_CLOSE_MAX_RADIUS
            ..Default::default()
        };
        let mask = refine_segmentation_mask(&image, &gap_mask, &[large_font_block.clone()]);
        assert_eq!(
            mask.get_pixel(25, 25)[0],
            255,
            "large-font gap should be bridged"
        );

        // The same absolute gap next to a tiny-font block should NOT be
        // bridged: at that scale a gap this size relative to the glyph is
        // more likely real background than an under-predicted pixel, and
        // a fixed "always aggressive" radius would incorrectly erase it.
        let small_font_block = TextRegion {
            detected_font_size_px: Some(10.0), // clamps to HOLE_CLOSE_MIN_RADIUS
            ..large_font_block
        };
        let mask = refine_segmentation_mask(&image, &gap_mask, &[small_font_block]);
        assert_eq!(
            mask.get_pixel(25, 25)[0],
            0,
            "small-font gap should stay open"
        );
    }

    #[test]
    fn refine_segmentation_mask_clips_outside_blocks() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([255, 255, 255])));
        let pred_mask = GrayImage::from_fn(32, 32, |x, y| {
            if (8..24).contains(&x) && (10..22).contains(&y) {
                Luma([200])
            } else {
                Luma([0])
            }
        });

        // No `detector` set, so this exercises the default (e.g.
        // pp-doclayout-v3) path, not the old "ctd" special case.
        let block = TextRegion {
            x: 10.0,
            y: 11.0,
            width: 4.0, // raw bbox roughly [10, 11] to [14, 15]
            height: 4.0,
            detected_font_size_px: Some(4.0), // pads by ~2px on each side
            ..Default::default()
        };

        let mask = refine_segmentation_mask(&image, &pred_mask, &[block]);
        let without_blocks = refine_segmentation_mask(&image, &pred_mask, &[]);

        // Assert providing bounding blocks saves the mask within bounds
        assert_ne!(mask, without_blocks);
        // Assert pixel INSIDE the block is preserved
        assert_eq!(mask.get_pixel(12, 13)[0], 255);
        // Assert pixel well outside the block, beyond even the padding, is cleared
        assert_eq!(mask.get_pixel(20, 13)[0], 0);
        // A pixel just outside the RAW detector box, but within the small
        // font-scaled padding, is preserved. Every detector's box gets
        // this padding now, not just "ctd" — any detector's box can be
        // off by a couple pixels, and a zero-slack clip would silently
        // re-exclude real glyph pixels no matter how good the mask or
        // gap-closing step is.
        assert_eq!(mask.get_pixel(15, 13)[0], 255);
    }

    #[test]
    fn extract_text_block_regions_falls_back_to_bbox_without_lines() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(24, 24, Rgb([255, 255, 255])));
        let block = TextRegion {
            x: 4.0,
            y: 5.0,
            width: 10.0,
            height: 8.0,
            ..Default::default()
        };

        // No detector/line_polygons set, so this crop now also gets the
        // small font-scaled padding (font falls back to min(width, height)
        // = 8 here), rather than the exact raw box: every detector's crop
        // gets a little slack now, not just "ctd".
        let regions = extract_text_block_regions(&image, &block);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].width(), 14);
        assert_eq!(regions[0].height(), 12);
    }

    #[test]
    fn crop_text_block_bbox_expands_ctd_crop() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(48, 48, Rgb([255, 255, 255])));
        let block = TextRegion {
            x: 10.0,
            y: 12.0,
            width: 12.0,
            height: 8.0,
            line_polygons: Some(vec![[
                [10.0, 12.0],
                [22.0, 12.0],
                [22.0, 20.0],
                [10.0, 20.0],
            ]]),
            source_direction: Some(TextDirection::Horizontal),
            rotation_deg: Some(0.0),
            detected_font_size_px: Some(8.0),
            detector: Some("ctd".to_string()),
            ..Default::default()
        };

        let crop = crop_text_block_bbox(&image, &block);
        assert!(crop.width() > 12);
        assert!(crop.height() > 8);
    }
}
