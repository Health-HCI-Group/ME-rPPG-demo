/*
* main.js
* The pipeline and MediaPipe processing (MediaPipe does not work in WebWorkers)
* */

// document.getElementById("authorization").addEventListener("click", () => {
//     navigator.mediaDevices.getUserMedia({
//         video: true,
//     }).then((stream) => {
//         if (stream) {
//             stream.getTracks().forEach(track => track.stop());
//             location.reload();
//         }
//     });
// });

// const cameraSelect = document.getElementById("cameraSelectDropdown");

// async function getCameraList() {
//     const devices = await navigator.mediaDevices.enumerateDevices();
//     const videoDevices = devices.filter(device => device.kind === "videoinput");
//     cameraSelect.options.length = 0;
//     videoDevices.forEach((device) => {
//         cameraSelect.options.add(new Option(device.label, device.deviceId));
//     });
// }
//
// getCameraList().then(() => {
//     cameraReady = true;
//     cameraButton.disabled = !ready();
// });

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
heartRateValue.style.color = "blue";
const inferenceFpsValue = document.getElementById("inferenceFpsValue");
const lambdaValue = document.getElementById("lambdaValue");
const lambdaValueDisplay = document.getElementById("lambdaValueDisplay");

let lambda = 1;

lambdaValue.addEventListener("input", (event) => {
    lambda = Math.pow(10, parseFloat(event.target.value));
    lambdaValueDisplay.innerText = `${lambda.toFixed(3)}`;
});

let video = null;

// let cameraReady = false;
let modelReady = false;
let stateReady = false;
let welchReady = false;
let hrReady = false;
// const ready = () => cameraReady && modelReady && stateReady && welchReady && hrReady;
const ready = () => modelReady && stateReady && welchReady && hrReady;

const isApplePlatform = () => {
    const isApple = /iPad|iPhone|iPod|Macintosh/.test(navigator.userAgent) && !window.MSStream;
    const supportsRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
    return isApple || !supportsRVFC;
};

//const isApplePlatform = () => true

import { FaceDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.4";

class KalmanFilter1D {
    constructor(processNoise, measurementNoise, initialState, initialEstimateError) {
        this.processNoise = processNoise;
        this.measurementNoise = measurementNoise;
        this.estimate = initialState;
        this.estimateError = initialEstimateError;
    }

    update(measurement) {
        const prediction = this.estimate;
        const predictionError = this.estimateError + this.processNoise;
        const kalmanGain = predictionError / (predictionError + this.measurementNoise);
        this.estimate = prediction + kalmanGain * (measurement - prediction);
        this.estimateError = (1 - kalmanGain) * predictionError;

        return this.estimate;
    }
}

let kfOriginX = null;
let kfOriginY = null;
let kfWidth = null;
let kfHeight = null;
let kfOutput = null;
let kfHr = null;


let faceDetector = null;

async function initializeFaceDetector() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.4/wasm"
    );
    faceDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `blaze_face_short_range.tflite`,
            delegate: "CPU",
        },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.5,
    });
}

let stream = null;
let isCameraOn = false;
let rafHandle = null;

let timestampArray = [];

async function initFaceDetector() {
    if (!faceDetector){
        await initializeFaceDetector();
    }
    console.log(`Face Detector OK state: ${!!faceDetector}`);
}

initFaceDetector().then();

// function startCamera() {
//     try {
//         dropCount = 30;
//         navigator.mediaDevices.getUserMedia({
//             video: {
//                 deviceId: {
//                     exact: cameraSelect.value ?? "default",
//                 },
//                 width: {
//                     ideal: 640,
//                 },
//                 height: {
//                     ideal: 480,
//                 },
//                 frameRate: 30,
//             },
//             audio: false,
//         }).then((mediaStream) => {
//             stream = mediaStream;
//             video.srcObject = stream;
//             isCameraOn = true;
//             cameraButton.textContent = "Stop";
//             rafHandle = video.requestVideoFrameCallback(processFrame);
//         });
//     } catch (error) {
//         console.error("Error accessing camera:", error);
//         alert("Unable to start camera");
//     }
// }


function stopCamera() {
    console.log("stop");
    isCameraOn = false;
    if (stream) {
        stream.getTracks().forEach(track => {
            track.stop();
            track.enabled = false;
        });
        stream = null;
    }
    video.pause();
    plotWorker.postMessage({output: null});
    video.cancelVideoFrameCallback(rafHandle);
    if (stream) {
        stream.getTracks().forEach((track) => {
            track.stop();
            track.enabled = false;
        });
    }
    video.srcObject = null;
    cameraButton.textContent = "Start";
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    const overlayCtx = overlayCanvas.getContext("2d");
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    timestampArray = [];
}

function startFrameProcessing() {
    if (!isApplePlatform()) {
        rafHandle = video.requestVideoFrameCallback(processFrame);
    } else {
        let lastCall = 0;
        const fpsInterval = 1000 / 30; //以最高30fps的速率轮询

        const animate = (now) => {
            if (!isCameraOn) return;

            const elapsed = now - lastCall;
            if (elapsed > fpsInterval) {
                lastCall = now - (elapsed % fpsInterval);

                const metadata = {
                    mediaTime: video.currentTime,
                    presentedFrames: 0,
                    width: video.videoWidth,
                    height: video.videoHeight
                };
                processFrame(performance.now(), metadata).then();
            }
            requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
    }
}

async function toggleCamera() {
    if (isCameraOn) {
        stopCamera();
        return;
    }
    try {
        // 同步执行链
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 },
                facingMode: "user"
            },
            audio: false,
        });

        // 确保视频元素已初始化
        if (!video) {
            video = document.getElementById("videoInput");
            video.addEventListener('loadedmetadata', () => {
                video.currentTime = 0.001;
                previewCanvas.width = video.videoWidth;
                previewCanvas.height = video.videoHeight;
                overlayCanvas.width = video.videoWidth;
                overlayCanvas.height = video.videoHeight;
            });
        }

        // 同步设置媒体流
        video.srcObject = stream;

        // iOS 必须的播放触发
        await video.play().then(() => {
            console.log('视频播放成功，当前状态:',
                `paused: ${video.paused}, `,
                `readyState: ${video.readyState}, `,
                `error: ${video.error}`
            );
            startFrameProcessing();
        }).catch(err => {
            console.error('video.play()拒绝:', {
                name: err.name,
                message: err.message,
                stack: err.stack
            });
            throw err;
        });

        // 启动后续处理
        isCameraOn = true;
        cameraButton.textContent = "Stop";
        // rafHandle = video.requestVideoFrameCallback(processFrame);

    } catch (error) {
        console.error('摄像头错误:', error.name, error.message);
        alert(`摄像头启动失败: ${error.message}`);
        stopCamera();
    }
}

//cameraButton.addEventListener("click", toggleCamera);

cameraButton.addEventListener("click", () => {
        if (!faceDetector) {
            initFaceDetector().then();
        }
        toggleCamera().then();
     });

const onnxWorker = new Worker("onnxWorker.js");
const plotWorker = new Worker("plotWorker.js");
const welchWorker = new Worker("welchWorker.js");

let welchArray = new Array(300).fill(0);
let welchCount = 300-90;

let inferenceTimestamp = 0;
let inferenceCount = 0;
let inputQueueCount = 0;
let dropCount = 30;

onnxWorker.onmessage = (event) => {
    const { type } = event.data;
    if (type === "ready") {
        const { which } = event.data;
        switch (which) {
            case "model":
                modelReady = true;
                break;
            case "state":
                stateReady = true;
                break;
        }
        cameraButton.disabled = !ready();
        return;
    }
    inputQueueCount--;
    const { output, delay, timestamp } = event.data;
    if (dropCount) return dropCount--;
    if (!kfOutput) {
        const processNoise = 1;
        const measurementNoise = 0.5;
        kfOutput = new KalmanFilter1D(processNoise, measurementNoise, output, 1);
    } else {
        kfOutput.update(output);
    }
    inferenceCount++;
    if (inferenceCount === 30) {
        inferenceFpsValue.textContent = `${(30 / ((timestamp - inferenceTimestamp) / 1000)).toFixed(1)}`;
        inferenceTimestamp = timestamp;
        inferenceCount = 0;
    }
    inferenceDelayValue.textContent = delay;
    if (welchArray.length >= 300) {
        welchArray.shift();
    }
    welchArray.push(kfOutput.estimate);
    welchCount++;
    plotWorker.postMessage({
        output:kfOutput.estimate,
    });
    if (welchCount >= 300) {
        welchWorker.postMessage({input: new Float32Array(welchArray)});
        welchCount = 270;
    }
}

plotWorker.onmessage = (event) => {
    const { imageBitmap } = event.data;
    plotCtx.transferFromImageBitmap(imageBitmap);
}

let MeanHRErr = 0.04;

welchWorker.onmessage = (event) => {
    const { type } = event.data;
    if (type === "ready") {
        const { which } = event.data;
        switch (which) {
            case "welch":
                welchReady = true;
                break;
            case "hr":
                hrReady = true;
                break;
        }
        cameraButton.disabled = !ready();
        return;
    }
    let { hr } = event.data;
    if (timestampArray.length > 300){
        const recentTimestamps = timestampArray.slice(-301);

        let totalDuration = 0;
        let validIntervals = 0;

        for (let i = 1; i < recentTimestamps.length; i++) {
            const delta = recentTimestamps[i] - recentTimestamps[i - 1];
            
            if (delta <= 0.5) {
                totalDuration += delta;
                validIntervals++;
            }
        }

        const averageFps = totalDuration > 0 
            ? (validIntervals / totalDuration)
            : 0;

        console.log(averageFps)
        hr = (hr / 30) * averageFps;
    } else {
        heartRateValue.style.color = "blue";
    }
    if (!kfHr){
        const processNoise = 1.;
        const measurementNoise = 2.;
        kfHr = new KalmanFilter1D(processNoise, measurementNoise, hr, 1);
    } else {
        kfHr.update(hr);
    }
    MeanHRErr = 0.8 * MeanHRErr + 0.2 * Math.abs(kfHr.estimate - hr) / hr;
    //console.log(MeanHRErr);
    if ((MeanHRErr < 0.02) && (heartRateValue.style.color === "blue")){
        heartRateValue.style.color = "red";
    }
    if ((MeanHRErr > 0.025) && (heartRateValue.style.color === "red")){
        heartRateValue.style.color = "blue";
    }
    heartRateValue.textContent = kfHr.estimate.toFixed(1);
    console.log(kfHr.estimate.toFixed(1));
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
let mediaTime = 0;
async function processFrame(now, metadata) {
    // lastTime = metadata.mediaTime;
    if (!isCameraOn) {
        console.log('Camera not on');
        return;
    }
    //console.log('New frame', lastTime, metadata.mediaTime);
    if (isApplePlatform()) {
        mediaTime = now / 1000;
    } else {
        //lastTime = metadata.mediaTime;
        mediaTime = performance.now()/1000
    }
    lastTime = mediaTime;
    if (inputQueueCount<5){
        //console.log(lastTime)
        timestampArray.push(lastTime);
        if (timestampArray.length > 301) {
            timestampArray.shift();
        }
        previewCtx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);

        if (!faceDetector) return;

        const startTimeMs = performance.now();
        const result = faceDetector.detectForVideo(video, startTimeMs);
        const detections = result.detections;

        if (detections && detections.length > 0) {
            const detection = detections[0];
            const rawBoundingBox = detection.boundingBox;
            if (!kfOriginX){
                const processNoise = 1e-2;
                const measurementNoise = 5e-1;
                kfOriginX = new KalmanFilter1D(processNoise, measurementNoise, rawBoundingBox.originX, 1);
                kfOriginY = new KalmanFilter1D(processNoise, measurementNoise, rawBoundingBox.originY, 1);
                kfWidth = new KalmanFilter1D(processNoise, measurementNoise, rawBoundingBox.width, 1);
                kfHeight = new KalmanFilter1D(processNoise, measurementNoise, rawBoundingBox.height, 1);
            }else{
                kfOriginX.update(rawBoundingBox.originX);
                kfOriginY.update(rawBoundingBox.originY);
                kfWidth.update(rawBoundingBox.width);
                kfHeight.update(rawBoundingBox.height);
            }
            const filteredBoundingBox = {
                originX: kfOriginX.estimate,
                originY: kfOriginY.estimate,
                width: kfWidth.estimate,
                height: kfHeight.estimate
            };
            filteredBoundingBox.height *= 1.2;
            filteredBoundingBox.originY -= filteredBoundingBox.height * 0.2;
            detection.boundingBox = filteredBoundingBox;
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
            inputQueueCount += 1;
            onnxWorker.postMessage({ type: "data", input, timestamp: lastTime, lambda });
        }
    }
    if (!isApplePlatform()){
        rafHandle = video.requestVideoFrameCallback(processFrame);
    }
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
