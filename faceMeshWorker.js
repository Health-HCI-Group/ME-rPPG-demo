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
        self.postMessage({landmarks});
    }
});

self.onmessage = (event) => {
    const { image } = event.data;
    faceMesh.send({image});
}
