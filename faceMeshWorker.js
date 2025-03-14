importScripts("https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js");

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
        self.postMessage({input});
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

self.onmessage = (event) => {
    const { image } = event.data;
    faceMesh.send({image});
}
