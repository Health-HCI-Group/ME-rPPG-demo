importScripts("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.cjs");

let vision;
let faceLandmarker;

FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
).then((v) => {vision = v;});

FaceLandmarker.createFromOptions(
    vision,
    {
        baseOptions: {
            modelAssetPath: "./",
            delegate: "CPU",
        },
    },
).then((landmarker) => {
    faceLandmarker = landmarker;
    console.log("Face Landmarker loaded");
});

self.onmessage = (event) => {
    const beginTime = Date.now();
    const { image } = event.data;
    const results = faceLandmarker.detect(image);
    const delay = Date.now() - beginTime;
    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        self.postMessage({landmarks, delay});
    }
}
