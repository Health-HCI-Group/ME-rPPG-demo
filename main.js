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

video.addEventListener('loadedmetadata', () => {
    previewCanvas.width = video.videoWidth;
    previewCanvas.height = video.videoHeight;
});

let stream = null;
let isCameraOn = false;

async function startCamera() {
    try {
        console.log(`Face Mesh OK state: ${!!FaceMesh}`);
        console.log(`faceMesh instance state: ${!!faceMesh}`);
        stream = await navigator.mediaDevices.getUserMedia({
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
        });
        video.srcObject = stream;
        video.play();
        isCameraOn = true;
        cameraButton.textContent = "Stop Camera";
        video.requestVideoFrameCallback(processFrame);
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
        startCamera().then();
    }
}

cameraButton.addEventListener("click", toggleCamera);

const onnxWorker = new Worker("onnxWorker.js");
const plotWorker = new Worker("plotWorker.js");
const welchWorker = new Worker("welchWorker.js");

let welchArray = [];
let welchCount = 0;

onnxWorker.onmessage = (event) => {
    const { output, delay } = event.data;
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
    console.log(`HR: ${hr}`);
}

const faceMesh = new FaceMesh({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    }
});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
});

faceMesh.onResults(async (results) => {
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];
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
});

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

let lastTime = 0;

async function processFrame(now, metadata) {
    if (!isCameraOn) return;
    if (metadata.mediaTime !== lastTime) {
        lastTime = metadata.mediaTime;
        previewCtx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
        await faceMesh.send({image: video});
    }
    console.log(`New Frame`)
    video.requestVideoFrameCallback(processFrame);
}
