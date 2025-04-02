importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js");

let onnxSession;
let state = {};

let lastTimestamp = null;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

ort.InferenceSession.create("model.onnx", {
    executionProviders: ["wasm"],
}).then((session) => {
    onnxSession = session;
    console.log("Model Session created");
});

function shapeOf(array) {
    const shape = [];
    let current = array;
    while (Array.isArray(current)) {
        shape.push(current.length);
        current = current[0];
    }
    return shape;
}

fetch("./state.json")
    .then((res) => res.json())
    .then((data) => {
        for (const [key, value] of Object.entries(data)) {
            const shape = shapeOf(value);
            const array = new Float32Array(value.flat(Infinity));
            state[key] = new ort.Tensor("float32", array, shape);
        }
        console.log("Initial state loaded");
    })

self.onmessage = async (event) => {
    if (!onnxSession || !state) {
        console.log("Model session or state not ready");
        return;
    }
    const startTime = Date.now();
    const { input, timestamp } = event.data;
    const inputData = new ort.Tensor("float32", input, [1, 1, 36, 36, 3]);
    const dt = new ort.Tensor("float32", [Math.max((lastTimestamp ? (timestamp - lastTimestamp)*2 : 1 / 30), 1/90)], []);
    lastTimestamp = timestamp;
    const feeds = {};
    feeds[onnxSession.inputNames[0]] = inputData;
    for (const [key, value] of Object.entries(state)) {
        feeds[key] = value;
    }
    feeds[onnxSession.inputNames[37]] = dt;
    const outputs = await onnxSession.run(feeds);
    const output = outputs[onnxSession.outputNames[0]]["cpuData"]["0"];
    for (let i = 1; i < onnxSession.outputNames.length; i++) {
        state[onnxSession.inputNames[i]] = outputs[onnxSession.outputNames[i]];
    }
    const nowTime = Date.now();
    const delay = nowTime - startTime;
    self.postMessage({output, delay, timestamp: nowTime});
};
