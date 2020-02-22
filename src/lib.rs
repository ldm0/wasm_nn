mod nn;
mod data;

use std::mem;
use std::slice;
//use std::os::raw::{/*c_double, c_int, */c_void};    // for js functions imports
use std::sync::Mutex;                   // for lazy_static
use lazy_static::lazy_static;           // for global variables

use ndarray::prelude::*;
use ndarray::{Array, Array1, Array3, Axis, array, Zip};

use nn::Network;
use data::Data;

#[derive(Default)]
struct MetaData {
    fc_size: u32,
    num_classes: u32,
    descent_rate: f32,
    regular_rate: f32
}

#[derive(Default)]
struct CriticalSection(MetaData, Data, Network);

// Imported js functions
extern "C" {
    // for debug
    fn log_u64(num: u32);
    // for data pointer draw
    // x,y: the offset from upper left corner
    // label: a fractal which represents the position current label is in total
    // position range
    fn draw_point(x: u32, y: u32, label_ratio: f32);
}

lazy_static! {
    static ref DATA: Mutex<CriticalSection> = Mutex::default();
}

#[no_mangle]
// This function returns the offset of the allocated buffer in wasm memory
pub fn alloc(size: u32) -> *mut u8 {
    let mut buffer: Vec<u8> = Vec::with_capacity(size as usize);
    let buffer_ptr = buffer.as_mut_ptr();
    mem::forget(buffer);
    buffer_ptr
}

#[no_mangle]
pub fn free(buffer_ptr: *mut u8, size: u32) {
    let _  = unsafe { Vec::from_raw_parts(buffer_ptr, 0, size as usize) };
}

#[no_mangle]
pub fn init(
    data_radius: f32,
    data_spin_span: f32,
    data_num: u32,
    num_classes: u32,
    data_gen_rand_max: f32,
    network_gen_rand_max: f32,
    fc_size: u32,
    descent_rate: f32,
    regular_rate: f32
) {
    // Thanks rust compiler :-/
    let ref mut tmp = *DATA.lock().unwrap();
    let CriticalSection(metadata, data, network) = tmp;

    metadata.fc_size        = fc_size;
    metadata.num_classes    = num_classes;
    metadata.descent_rate   = descent_rate;
    metadata.regular_rate   = regular_rate;

    // Num of each data class is the same 
    data.init(num_classes, data_num / num_classes, data_radius, data_spin_span, data_gen_rand_max);

    // Input of this network is two dimension points
    // output label is sparsed num_classes integer
    const PLANE_DIMENSION: u32 = 2;
    network.init(PLANE_DIMENSION, fc_size, num_classes, network_gen_rand_max);
}

#[no_mangle]
pub fn train() -> f32 {
    let ref mut tmp = *DATA.lock().unwrap();
    // Jesus, thats magic 
    let CriticalSection(ref metadata, ref data, ref mut network) = *tmp;

    let regular_rate = metadata.regular_rate;
    let descent_rate = metadata.descent_rate;

    let (fc_layer, softmax) = network.forward_propagation(&data.points);
    let (dw1, db1, dw2, db2) = network.back_propagation(&data.points, &fc_layer, &softmax, &data.labels, regular_rate);
    let loss = network.loss(&softmax, &data.labels, regular_rate);
    network.descent(&dw1, &db1, &dw2, &db2, descent_rate);

    let (data_loss, regular_loss) = loss;
    data_loss + regular_loss
}

// Plot classified backgroud to canvas
// span_least The least span of area should be drawn to canvas(because usually the canvas is not square)
#[no_mangle]
pub fn draw_prediction(canvas: *mut u8, width: u32, height: u32, span_least: f32) {
    // assert!(span_least > 0f32);
    let width = width as usize;
    let height = height as usize;

    // `data` will be used to draw data points
    let ref tmp = *DATA.lock().unwrap();
    let CriticalSection(metadata, _, network) = tmp;

    let num_classes = metadata.num_classes as usize;

    let r: Array1<f32> = Array::linspace(0f32, 200f32, num_classes);
    let g: Array1<f32> = Array::linspace(0f32, 240f32, num_classes);
    let b: Array1<f32> = Array::linspace(0f32, 255f32, num_classes);

    let span_per_pixel = span_least / width.min(height) as f32;
    let span_height = height as f32 * span_per_pixel;
    let span_width = width as f32 * span_per_pixel;

    let width_max = span_width / 2f32;
    let width_min = -span_width / 2f32;
    let height_max = span_height / 2f32;
    let height_min = -span_height / 2f32;

    let x_axis: Array1<f32> = Array::linspace(width_min, width_max, width);
    let y_axis: Array1<f32> = Array::linspace(height_min, height_max, height);

    // coordination 
    let mut grid: Array3<f32> = Array::zeros((height, width, 2));
    for y in 0..height {
        for x in 0..width {
            let coord = array![x_axis[[x]], y_axis[[y]]];
            let mut slice = grid.slice_mut(s![y, x, ..]);
            slice.assign(&coord);
        }
    }

    let xys = grid.into_shape((height * width, 2)).unwrap();
    let (_, softmax) = network.forward_propagation(&xys);
    let mut labels: Array1<usize> = Array::zeros(height * width);
    for (y, row) in softmax.axis_iter(Axis(0)).enumerate() {
        let mut maxx = 0 as usize;
        let mut max = row[[0]];
        for (x, col) in row.iter().enumerate() {
            if *col > max {
                maxx = x;
                max = *col;
            }
        }
        labels[[y]] = maxx;
    }
    let grid_label = labels.into_shape((height, width)).unwrap();

    let canvas_size = width * height * 4;
    let canvas: &mut [u8] = unsafe{slice::from_raw_parts_mut(canvas, canvas_size)};
    for y in 0..height {
        for x in 0..width {
            // assume rgba
            canvas[4 * (y * width + x) + 0] = r[[grid_label[[y, x]]]] as u8;
            canvas[4 * (y * width + x) + 1] = g[[grid_label[[y, x]]]] as u8;
            canvas[4 * (y * width + x) + 2] = b[[grid_label[[y, x]]]] as u8;
            canvas[4 * (y * width + x) + 3] = 0xFF as u8;
        }
    }
}

// check parameters for function below which draws predictions
#[no_mangle]
pub fn draw_points(width: u32, height: u32, span_least: f32) {
    let ref tmp = *DATA.lock().unwrap();
    let CriticalSection(metadata, data, _) = tmp;
    let num_classes = metadata.num_classes as f32;

    let pixel_per_span = width.min(height) as f32 / span_least;
    let labels = &data.labels;
    let points = &data.points;
    let points_x = points.index_axis(Axis(1), 0);
    let points_y = points.index_axis(Axis(1), 1);
    Zip::from(labels)
        .and(points_x)
        .and(points_y)
        .apply(|&label, &x, &y| {
            // Assume data position is limited in:
            // [-data_radius - data_rand_max, data_radius + data_rand_max]
            let x = (x * pixel_per_span) as i64 + width as i64 / 2;
            let y = (y * pixel_per_span) as i64 + height as i64 / 2;

            // if points can show in canvas
            if !(x >= width as i64 || x < 0 || y >= height as i64 || y < 0) {
                // floor
                let x = x as u32;
                let y = y as u32;
                let label_ratio = label as f32 / num_classes;
                unsafe { draw_point(x, y, label_ratio); }
            }
        });
}
 
#[cfg(test)]
mod kernel_test {
    use super::*;

    lazy_static! {
        static ref POINT_DRAW_TIMES: Mutex<u32> = Mutex::new(0);
    }

    // Override the extern functions
    #[no_mangle]
    extern "C" fn draw_point(_: u32, _: u32, _: f32) {
        *POINT_DRAW_TIMES.lock().unwrap() += 1;
    }

    use std::f32::consts::PI;               // for math functions

    const DATA_GEN_RADIUS: f32 = 1f32;
    const SPIN_SPAN: f32 = PI;
    const NUM_CLASSES: u32 = 3;
    const DATA_NUM: u32 = 300;
    const FC_SIZE: u32 = 100;
    const REGULAR_RATE: f32 = 0.001f32;
    const DESCENT_RATE: f32 = 1f32;
    const DATA_GEN_RAND_MAX: f32 = 0.25f32;
    const NETWORK_GEN_RAND_MAX: f32 = 0.1f32;

    #[test]
    fn test_all() {
        init(DATA_GEN_RADIUS, SPIN_SPAN, DATA_NUM, NUM_CLASSES, DATA_GEN_RAND_MAX, NETWORK_GEN_RAND_MAX, FC_SIZE, DESCENT_RATE, REGULAR_RATE);
        let loss_before: f32 = train();
        for _ in 0..50 {
            let loss = train();
            assert!(loss < loss_before * 1.1f32);
        }
    }

    #[test]
    fn test_buffer_allocation() {
        let buffer = alloc(114514);
        free(buffer, 114514);
    }

    #[test]
    fn test_draw_prediction() {
        init(DATA_GEN_RADIUS, SPIN_SPAN, DATA_NUM, NUM_CLASSES, DATA_GEN_RAND_MAX, NETWORK_GEN_RAND_MAX, FC_SIZE, DESCENT_RATE, REGULAR_RATE);
        let width = 100;
        let height = 100;
        let buffer = alloc(width * height * 4);
        draw_prediction(buffer, width, height, 2f32);
        free(buffer, width * height * 4);
    }

    #[test]
    fn test_draw_points() {
        // Because cargo test is default multi-thread, put them together to avoid data_racing

        // span_least * 1.1 for padding

        init(DATA_GEN_RADIUS, SPIN_SPAN, DATA_NUM, NUM_CLASSES, DATA_GEN_RAND_MAX, NETWORK_GEN_RAND_MAX, FC_SIZE, DESCENT_RATE, REGULAR_RATE);

        // test small resolution drawing
        *POINT_DRAW_TIMES.lock().unwrap() = 0;
        draw_points(1, 1, DATA_GEN_RADIUS * 2f32 * 1.1f32);
        assert_eq!(DATA_NUM, *POINT_DRAW_TIMES.lock().unwrap());

        // test tall screen drawing
        *POINT_DRAW_TIMES.lock().unwrap() = 0;
        draw_points(1, 100, DATA_GEN_RADIUS * 2f32 * 1.1f32);
        assert_eq!(DATA_NUM, *POINT_DRAW_TIMES.lock().unwrap());

        // test flat screen drawing
        *POINT_DRAW_TIMES.lock().unwrap() = 0;
        draw_points(1, 100, DATA_GEN_RADIUS * 2f32 * 1.1f32);
        assert_eq!(DATA_NUM, *POINT_DRAW_TIMES.lock().unwrap());

        // test square screen drawing
        *POINT_DRAW_TIMES.lock().unwrap() = 0;
        draw_points(100, 100, DATA_GEN_RADIUS * 2f32 * 1.1f32);
        assert_eq!(DATA_NUM, *POINT_DRAW_TIMES.lock().unwrap());

        // test huge screen drawing
        *POINT_DRAW_TIMES.lock().unwrap() = 0;
        draw_points(10000000, 1000000, DATA_GEN_RADIUS * 2f32 * 1.1f32);
        assert_eq!(DATA_NUM, *POINT_DRAW_TIMES.lock().unwrap());
    }
}
