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

const faceMeshWorker = new Worker("faceMeshWorker.js");
const onnxWorker = new Worker("onnxWorker.js");
const plotWorker = new Worker("plotWorker.js");
const welchWorker = new Worker("welchWorker.js");

let welchArray = [];
let welchCount = 0;

let inferenceTimestamp = 0;
let inferenceCount = 0;

faceMeshWorker.onmessage = (event) => {
    const { input } = event.data;
    onnxWorker.postMessage({input});
}

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
        faceMeshWorker.postMessage({image: previewCtx.getImageData(0, 0, previewCanvas.width, previewCanvas.height)});
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
