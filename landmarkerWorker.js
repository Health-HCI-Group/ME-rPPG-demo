// importScripts("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs");

import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

let vision;
let faceLandmarker;

FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
).then((v) => {
    vision = v;
    FaceLandmarker.createFromOptions(
        vision,
        {
            baseOptions: {
                modelAssetPath: "./face_landmarker.task",
                delegate: "CPU",
            },
            runningMode: "IMAGE",
            numFaces: 1,
        },
    ).then((landmarker) => {
        faceLandmarker = landmarker;
        console.log("Face Landmarker loaded");
    });
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
