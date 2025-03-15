/*
* main.js
* The pipeline and MediaPipe processing (MediaPipe does not work in WebWorkers)
* */

const cameraButton = document.getElementById("switchButton");
const previewCanvas = document.getElementById("previewCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const previewCtx = previewCanvas.getContext("2d");
const faceCanvas = document.createElement("canvas");
const faceCtx = faceCanvas.getContext("2d");
const plotCanvas = document.getElementById("plotCanvas");
const plotCtx = plotCanvas.getContext("bitmaprenderer");
const inferenceDelayValue = document.getElementById("inferenceDelayValue");
const heartRateValue = document.getElementById("heartRateValue");
heartRateValue.style.color = 'blue';
const inferenceFpsValue = document.getElementById("inferenceFpsValue");

const video = document.getElementById("videoInput");
video.addEventListener('loadedmetadata', () => {
    previewCanvas.width = video.videoWidth;
    previewCanvas.height = video.videoHeight;
    overlayCanvas.width = video.videoWidth;
    overlayCanvas.height = video.videoHeight;
});

import { FaceDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";


let faceDetector = null;

async function initializeFaceDetector() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  faceDetector = await FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    minDetectionConfidence: 0.5
  });
}


let stream = null;
let isCameraOn = false;
let rafHandle = null;

let timestampArray = [];

async function startCamera() {
    try {
        if (!faceDetector){
            await initializeFaceDetector();
        }
        console.log(`Face Detector OK state: ${!!faceDetector}`);
        
        navigator.mediaDevices.getUserMedia({
            video: {width: {ideal:640}, height: {ideal:480}, frameRate: 30},
            audio: false
        }).then((mediaStream) => {
            stream = mediaStream;
            video.srcObject = stream;
            isCameraOn = true;
            cameraButton.textContent = "Stop";
            rafHandle = video.requestVideoFrameCallback(processFrame);
        });
    } catch (error) {
        console.error("Error accessing camera:", error);
        alert("Unable to start camera");
    }
}


function stopCamera() {
    plotWorker.postMessage({output: null});

    video.cancelVideoFrameCallback(rafHandle);
    if (stream) {
        stream.getTracks().forEach((track) => {
            track.stop();
            track.enabled = false;
        });
    }
    video.srcObject = null;
    isCameraOn = false;
    cameraButton.textContent = "Start";
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    const overlayCtx = overlayCanvas.getContext("2d");
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    timestampArray = [];
}

function toggleCamera() {
    if (isCameraOn) {
        stopCamera();
    } else {
        startCamera();
    }
}

cameraButton.addEventListener("click", toggleCamera);

const onnxWorker = new Worker("onnxWorker.js");
const plotWorker = new Worker("plotWorker.js");
const welchWorker = new Worker("welchWorker.js");

let welchArray = [];
let welchCount = 0;

let inferenceTimestamp = 0;
let inferenceCount = 0;

onnxWorker.onmessage = (event) => {
    const { output, delay, timestamp } = event.data;
    inferenceCount++;
    if (inferenceCount === 30) {
        inferenceFpsValue.textContent = `${(30 / ((timestamp - inferenceTimestamp) / 1000)).toFixed(1)}`;
        inferenceTimestamp = timestamp;
        inferenceCount = 0;
    }
    inferenceDelayValue.textContent = delay;
    if (welchArray.length >= 150) {
        welchArray.shift();
    }
    welchArray.push(output);
    welchCount++;
    plotWorker.postMessage({
        output,
    });
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
    let { hr } = event.data;
    if (timestampArray.length > 150){
        const startTime = timestampArray[timestampArray.length - 151];
        const endTime = timestampArray[timestampArray.length - 1];
        const duration = endTime - startTime;
        const averageFps = 150 / duration;
        console.log(averageFps);
        hr = hr/30*averageFps;
        heartRateValue.style.color = 'red';
    }else{
        heartRateValue.style.color = 'blue';
    }
    heartRateValue.textContent = hr.toFixed(1);
}

function cropAndResizeUsingBoundingBox(canvas, boundingBox) {

    const x = Math.max(0, boundingBox.originX);
    const y = Math.max(0, boundingBox.originY);
    const width = Math.min(boundingBox.width, canvas.width - x);
    const height = Math.min(boundingBox.height, canvas.height - y);

    faceCanvas.width = width;
    faceCanvas.height = height;
    faceCtx.drawImage(canvas, x, y, width, height, 0, 0, width, height);

    const resizedCanvas = document.createElement("canvas");
    resizedCanvas.width = 36;
    resizedCanvas.height = 36;
    const resizedCtx = resizedCanvas.getContext("2d");
    resizedCtx.imageSmoothingEnabled = true;
    resizedCtx.imageSmoothingQuality = "high";
    resizedCtx.drawImage(faceCanvas, 0, 0, width, height, 0, 0, 36, 36);
    return resizedCanvas;
}


let lastTime = 0;

async function processFrame(now, metadata) {
    if (!isCameraOn) return;
    if (lastTime == metadata.mediaTime) return;
    lastTime = metadata.mediaTime;
    timestampArray.push(lastTime);
    previewCtx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);

    if (!faceDetector) return;
    
    const startTimeMs = performance.now();
    const result = faceDetector.detectForVideo(video, startTimeMs);
    const detections = result.detections;

    if (detections && detections.length > 0) {
        const detection = detections[0];
        detection.boundingBox.height = detection.boundingBox.height * 1.2;
        detection.boundingBox.originY = detection.boundingBox.originY - detection.boundingBox.height * 0.2;
        const faceImage = cropAndResizeUsingBoundingBox(previewCanvas, detection.boundingBox);
        drawBoundingBox(detection.boundingBox);
        const ctx = faceImage.getContext("2d");
        const imageData = ctx.getImageData(0, 0, 36, 36);
        const input = new Float32Array(36 * 36 * 3);
        
        for (let i = 0; i < imageData.data.length; i += 4) {
            const index = i / 4;
            input[index * 3] = imageData.data[i] / 255;
            input[index * 3 + 1] = imageData.data[i + 1] / 255;
            input[index * 3 + 2] = imageData.data[i + 2] / 255;
        }
        
        onnxWorker.postMessage({ input, timestamp: lastTime });
    }

    video.requestVideoFrameCallback(processFrame);
}

function drawBoundingBox(boundingBox) {

    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    ctx.strokeStyle = "#FF0000";
    ctx.lineWidth = 2;           
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.rect(
      boundingBox.originX,
      boundingBox.originY,
      boundingBox.width,
      boundingBox.height
    );
    ctx.stroke();
  }
  
  