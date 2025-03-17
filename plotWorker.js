let dataBuffer = [];
const bufferSize = 512;

function drawPlot() {
    // preparation
    const canvas = new OffscreenCanvas(640, 240);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#000000";
    ctx.fillStyle = "#000000";
    ctx.font = "20px Arial";
    const axisPadding = 50;
    const axisWidth = canvas.width - 2 * axisPadding;
    const axisHeight = canvas.height - 2 * axisPadding;

    // Bounding box
    ctx.beginPath();
    ctx.moveTo(axisPadding, axisPadding);
    ctx.lineTo(canvas.width - axisPadding, axisPadding);
    ctx.lineTo(canvas.width - axisPadding, canvas.height - axisPadding);
    ctx.lineTo(axisPadding, canvas.height - axisPadding);
    ctx.closePath();
    ctx.stroke();

    // Plot
    ctx.strokeStyle = "#FF3366";
    ctx.beginPath();
    const xStep = axisWidth / (bufferSize - 1);
    const yScale = axisHeight / 4;
    dataBuffer.forEach((value, index) => {
        const x = axisPadding + index * xStep;
        const y = canvas.height - axisPadding - (value + 2) * yScale;
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    // Convert to ImageBitmap and post
    canvas.convertToBlob().then((blob) => {
        createImageBitmap(blob).then((imageBitmap) => {
            self.postMessage({ imageBitmap });
        });
    });
}

self.onmessage = (event) => {
    const { output } = event.data;
    if (output === null) {
        dataBuffer = [];
        drawPlot();
        return;
    }
    if (dataBuffer.length >= bufferSize) {
        dataBuffer.shift();
    }
    dataBuffer.push(output);
    drawPlot();
};
