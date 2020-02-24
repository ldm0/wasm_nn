// Ratio: [0., 1.)
function hue(ratio) {
    let rgb = "";
    let hue = 6 * ratio;
    let integer_part = Math.floor(hue);
    let fractal_part = Math.round((hue - integer_part) * 255);
    switch (integer_part) {
    case 0: rgb = "#" + "FF" + zero_padding(fractal_part) + "00";          break;
    case 1: rgb = "#" + zero_padding(255 - fractal_part) + "FF" + "00";    break;
    case 2: rgb = "#" + "00" + "FF" + zero_padding(fractal_part);          break;
    case 3: rgb = "#" + "00" + zero_padding(255 - fractal_part) + "FF";    break;
    case 4: rgb = "#" + zero_padding(fractal_part) + "00" + "FF";          break;
    case 5: rgb = "#" + "FF" + "00" + zero_padding(255 - fractal_part);    break;
    }
    return rgb;
}

// Number to zero-padding hex string
function zero_padding(num) {
    let result = Math.round(num).toString(16);
    if (result.length < 2)
        result = "0" + result;
    return result;
}

async function main(){
    const canvas_width = 256;
    const canvas_height = 256;
    const canvas_buffer_size = canvas_width * canvas_height * 4;
    const data_span_radius = 1.;

    const canvas = document.getElementById("main_canvas");
    const canvas_context = canvas.getContext("2d");

    const control_button = document.getElementById("control_button");
    const loss_reveal = document.getElementById("loss_reveal");
    const input_data_spin_span = document.getElementById("input_data_spin_span");
    const input_data_num = document.getElementById("input_data_num");
    const input_num_classes = document.getElementById("input_num_classes");
    const input_data_gen_rand_max = document.getElementById("input_data_gen_rand_max");
    const input_network_gen_rand_max = document.getElementById("input_network_gen_rand_max");
    const input_fc_size = document.getElementById("input_fc_size");
    const input_descent_rate = document.getElementById("input_descent_rate");
    const input_regular_rate = document.getElementById("input_regular_rate");

    function get_settings() {
        let settings = [
            parseFloat(input_data_spin_span.value),
            parseInt(input_data_num.value),
            parseInt(input_num_classes.value),
            parseFloat(input_data_gen_rand_max.value),
            parseFloat(input_network_gen_rand_max.value),
            parseInt(input_fc_size.value),
            parseFloat(input_descent_rate.value),
            parseFloat(input_regular_rate.value),
        ];
        return settings;
    }

    function envs() {
        // For debug
        function log_u64(x) {
            console.log(x);
        }

        function draw_point(x, y, label) {
            canvas_context.beginPath();
            canvas_context.arc(x, y, 2, 0, 2 * Math.PI);
            canvas_context.fillStyle = hue(label) + "7f";
            canvas_context.fill();
        }
        let env = {
            log_u64,
            draw_point,
        };
        return env;
    }

    const kernel_stream = await fetch("../wasm/wasm_nn.wasm");
    const kernel = await WebAssembly.instantiateStreaming(kernel_stream, { env: envs()});
    
    const {alloc: kernel_alloc, free: kernel_free} = kernel.instance.exports;
    const {
        init: kernel_init,
        train: kernel_train,
        draw_prediction: kernel_draw_prediction,
        draw_points: kernel_draw_points
    } = kernel.instance.exports;
    const {memory} = kernel.instance.exports;

    // Alloc graphic buffer
    // Should not freed because you don't know when the drawing completes
    // Maybe not completed forever...
    //kernel_free(canvas_buffer_ptr, buffer_size);
    const canvas_buffer_ptr = kernel_alloc(canvas_buffer_size);

    function draw_frame() {
        // multiply 1.1 for spadding
        kernel_draw_prediction(canvas_buffer_ptr, canvas_width, canvas_height, data_span_radius * 2);
        const canvas_buffer_array = new Uint8ClampedArray(memory.buffer, canvas_buffer_ptr, canvas_buffer_size);
        const canvas_image_data = new ImageData(canvas_buffer_array, canvas_width, canvas_height)
        canvas_context.putImageData(canvas_image_data, 0, 0);

        kernel_draw_points(canvas_width, canvas_height, data_span_radius * 2);
    }

    function nninit(settings) {
        // Gen data for training. Check source code of kernel for parameter meaning
        kernel_init(
            data_span_radius,
            settings[0],
            settings[1],
            settings[2],
            settings[3],
            settings[4],
            settings[5],
            settings[6],
            settings[7],
        );
        // draw a fram to avoid blank canvas 
        draw_frame();
    }

    nninit(get_settings());

    {
        let run = false;

        {
            let counter = 0;
            function nnloop() {
                if (run) {
                    let loss = kernel_train();
                    if (counter >= 10) {
                        counter = 0;
                        loss_reveal.innerText = "loss: " + loss;
                        window.requestAnimationFrame(draw_frame);
                    }
                    setTimeout(nnloop, 0);
                    ++counter;
                }
            }
        }

        function nnstart() {
            run = true;
            nnloop();
        }

        function nnstop() {
            run = false;
        }
    }

    {
        let run = false;
        let current_settings = get_settings();

        control_button.onclick = () => {
            if (run) {
                run = false;
                control_button.innerText = "run";
                nnstop();
            } else {
                run = true;
                control_button.innerText = "stop";
                let new_settings = get_settings();
                if (JSON.stringify(current_settings) !== JSON.stringify(new_settings)) {
                    current_settings = new_settings;
                    nninit(current_settings);
                }
                nnstart();
            }
        }
    }
}

main();