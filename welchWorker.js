importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

let welchSession;
let hrSession;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

ort.InferenceSession.create("welch_psd.onnx", {
    executionProviders: ["wasm"],
}).then(async (session) => {
    welchSession = session;
    console.log("Welch Session created");
});

ort.InferenceSession.create("get_hr.onnx", {
    executionProviders: ["wasm"],
}).then((session) => {
    hrSession = session;
    console.log("HR Session created");
});

self.onmessage = async (event) => {
    if (!welchSession) {
        console.log("Welch session not ready");
        return;
    }
    const startTime = Date.now();
    const { input } = event.data;
    const inputData = new ort.Tensor("float32", input, [1, 1, input.length]);
    const outputs = await welchSession.run({ input: inputData });
    const freqs = outputs["freqs"];
    const psd = outputs["psd"];
    const hr = (await hrSession.run({freqs, psd}))["hr"]["cpuData"]["0"];
    self.postMessage({hr});
};
