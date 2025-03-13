importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

let welchSession;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

ort.InferenceSession.create("welch.onnx", {
    executionProviders: ["wasm"],
}).then((session) => {
    welchSession = session;
    console.log("Welch Session created");
});

self.onmessage = async (event) => {
    if (!welchSession) {
        console.log("Welch session not ready");
        return;
    }
    const startTime = Date.now();
    const { input } = event.data;
    const inputData = new ort.Tensor("float32", input, [1, input.length]);
    const outputs = await welchSession.run({ input: inputData });
};
