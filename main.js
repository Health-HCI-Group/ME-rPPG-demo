/*
* main.js
* The pipeline and MediaPipe processing (MediaPipe does not work in WebWorkers)
* */

const video = document.getElementById("videoInput");
const cameraButton = document.getElementById("switchButton");
const previewCanvas = document.getElementById("previewCanvas");
const previewCtx = previewCanvas.getContext("2d");
const faceCanvas = document.createElement("canvas");
const faceCtx = faceCanvas.getContext("2d");
const plotCanvas = document.getElementById("plotCanvas");
const plotCtx = plotCanvas.getContext("bitmaprenderer");
const inferenceDelayValue = document.getElementById("inferenceDelayValue");
const heartRateValue = document.getElementById("heartRateValue");
const cameraFpsValue = document.getElementById("cameraFpsValue");
const inferenceFpsValue = document.getElementById("inferenceFpsValue");

import mpTasksVision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

video.addEventListener('loadedmetadata', () => {
    previewCanvas.width = video.videoWidth;
    previewCanvas.height = video.videoHeight;
});

let stream = null;
let isCameraOn = false;

function startCamera() {
    try {
        navigator.mediaDevices.getUserMedia({
            video: {
                width: {
                    ideal: 640,
                },
                height: {
                    ideal: 480,
                },
                frameRate: 30,
            },
            audio: false,
        }).then((mediaStream) => {
            stream = mediaStream;
            video.srcObject = stream;
            video.play();
            isCameraOn = true;
            cameraButton.textContent = "Stop Camera";
            video.requestVideoFrameCallback(processFrame);
        });
    } catch (error) {
        console.error("Error accessing camera:", error);
        alert("Unable to start camera");
    }
}

function stopCamera() {
    plotWorker.postMessage({output: null});
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        video.srcObject = null;
        isCameraOn = false;
        cameraButton.textContent = "Start Camera";
    }
}

function toggleCamera() {
    if (isCameraOn) {
        stopCamera();
    } else {
        startCamera();
    }
}

cameraButton.addEventListener("click", toggleCamera);

// const landmarkerWorker = new Worker("landmarkerWorker.js", {type: "module"});
const onnxWorker = new Worker("onnxWorker.js");
const plotWorker = new Worker("plotWorker.js");
const welchWorker = new Worker("welchWorker.js");

let welchArray = [];
let welchCount = 0;

let inferenceTimestamp = 0;
let inferenceCount = 0;

// landmarkerWorker.onmessage = (event) => {
//     const { landmarks, delay } = event.data;
//     const faceImage = cropAndResize(previewCanvas, landmarks)
//     const ctx = faceImage.getContext("2d");
//     const imageData = ctx.getImageData(0, 0, faceImage.width, faceImage.height);
//     const input = new Float32Array(36 * 36 * 3);
//     for (let i = 0; i < imageData.data.length; i += 4) {
//         const index = i / 4;
//         input[index * 3] = imageData.data[i] / 255;
//         input[index * 3 + 1] = imageData.data[i + 1] / 255;
//         input[index * 3 + 2] = imageData.data[i + 2] / 255;
//     }
//     onnxWorker.postMessage({input});
// }

let vision;
let faceLandmarker;

mpTasksVision.FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
).then((v) => {
    vision = v;
    mpTasksVision.FaceLandmarker.createFromOptions(
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

onnxWorker.onmessage = (event) => {
    const { output, delay, timestamp } = event.data;
    inferenceCount++;
    if (inferenceCount === 30) {
        inferenceFpsValue.textContent = `${(30 / ((timestamp - inferenceTimestamp) / 1000)).toFixed(2)}`;
        inferenceTimestamp = timestamp;
        inferenceCount = 0;
    }
    inferenceDelayValue.textContent = `${delay}`;
    if (welchArray.length >= 150) {
        welchArray.shift();
    }
    welchArray.push(output);
    welchCount++;
    plotWorker.postMessage({output});
    if (welchCount >= 150) {
        welchWorker.postMessage({input: new Float32Array(welchArray)});
        welchCount = 120;
    }
}

plotWorker.onmessage = (event) => {
    const { imageBitmap } = event.data;
    plotCtx.transferFromImageBitmap(imageBitmap);
}

welchWorker.onmessage = (event) => {
    const { hr } = event.data;
    heartRateValue.textContent = hr.toFixed(1);
}

let lastTime = 0;
let frameCount = 0;
let fpsBeginTime = 0;

async function processFrame(now, metadata) {
    if (!isCameraOn) return;
    if (metadata.mediaTime !== lastTime) {
        lastTime = metadata.mediaTime;
        previewCtx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
        // landmarkerWorker.postMessage({image: await createImageBitmap(video)});
        const results = faceLandmarker.detect(video);
        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            const faceImage = cropAndResize(previewCanvas, landmarks)
            const ctx = faceImage.getContext("2d");
            const imageData = ctx.getImageData(0, 0, faceImage.width, faceImage.height);
            const input = new Float32Array(36 * 36 * 3);
            for (let i = 0; i < imageData.data.length; i += 4) {
                const index = i / 4;
                input[index * 3] = imageData.data[i] / 255;
                input[index * 3 + 1] = imageData.data[i + 1] / 255;
                input[index * 3 + 2] = imageData.data[i + 2] / 255;
            }
            onnxWorker.postMessage({input});
        }

        frameCount++;
        if (frameCount === 30) {
            const fpsEndTime = Date.now();
            const fpsInterval = fpsEndTime - fpsBeginTime;
            cameraFpsValue.textContent = `${(30 / (fpsInterval / 1000)).toFixed(2)}`;
            fpsBeginTime = fpsEndTime;
            frameCount = 0;
        }
    }
    video.requestVideoFrameCallback(processFrame);
}

function getLandmarksBounds(landmarks, width, height) {
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (let i = 0; i < landmarks.length; i++) {
        const { x, y } = landmarks[i];
        if (x < xMin) xMin = x;
        if (x > xMax) xMax = x;
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
    }
    return {
        xMin: Math.max(xMin, 0) * width,
        xMax: Math.min(xMax, 1) * width,
        yMin: Math.max(yMin, 0) * height,
        yMax: Math.min(yMax, 1) * height,
    }
}

function cropAndResize(canvas, landmarks) {
    const { xMin, xMax, yMin, yMax } = getLandmarksBounds(landmarks, canvas.width, canvas.height);
    const width = xMax - xMin;
    const height = yMax - yMin;
    faceCanvas.width = width;
    faceCanvas.height = height;
    faceCtx.drawImage(canvas, xMin, yMin, width, height, 0, 0, width, height);
    const resizedCanvas = document.createElement("canvas");
    resizedCanvas.width = 36;
    resizedCanvas.height = 36;
    const resizedCtx = resizedCanvas.getContext("2d");
    resizedCtx.drawImage(faceCanvas, 0, 0, width, height, 0, 0, 36, 36);
    return resizedCanvas;
}
