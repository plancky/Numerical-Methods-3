import { me } from "./dat.js";

const canvas = document.querySelector("#canvas");
const dimension = canvas.getBoundingClientRect();
const ctx = canvas.getContext("2d");
let graphPoints = [];
let circleStroke = 1.1;
let lineStroke = 1.3;
let speed = 1;
let skip = 1;
let time = 0;
let showCircle = true;

canvas.height =  window.innerHeight ;
canvas.width = window.innerWidth;

class Complex {
  constructor(x, y) {
    this.re = x;
    this.im = y;
  }

  add(other) {
    const re = this.re + other.re;
    const im = this.im + other.im;
    return new Complex(re, im);
  }

  multiply(other) {
    const re = this.re * other.re - this.im * other.im;
    const im = this.re * other.im + this.im * other.re;
    return new Complex(re, im);
  }

  devide(other) {
    const re = (this.re * other.re + this.im * other.im) / (other.re * other.re + other.im * other.im);
    const im = (this.im * other.re - this.re * other.im) / (other.re * other.re + other.im * other.im);
    return new Complex(re, im);
  }

  amplitude() {
    return Math.sqrt(this.re * this.re + this.im * this.im);
  }

  phase() {
    return Math.atan2(this.im, this.re);
  }
}

const clearCanvas = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
};

const drawCircle = (x, y, r, strokeWidth = 1, s = false) => {
  ctx.beginPath();
  ctx.lineWidth = strokeWidth * 0.2;
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  if (s) {
    ctx.fillStyle = "#fff";
    ctx.fill();
    return;
  }
  ctx.strokeStyle = "#fff";
  ctx.stroke();
};
  
const drawLine = (x1, y1, x2, y2, stroke = 1, color = "#fff") => {
  ctx.beginPath();
  ctx.lineWidth = stroke;
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.stroke();
};
  
const drawCurve = (pointsArr) => {
  for (let i = 0; i < pointsArr.length; i++) {
    if (pointsArr[i + 1])
      drawLine(pointsArr[i].x, pointsArr[i].y, pointsArr[i + 1].x, pointsArr[i + 1].y, lineStroke, "#00ffdd");
  }
};


const animate = (fun) => {
  for (let i = 0; i < speed; i+=100) {
    clearCanvas();
    fun();
  }
  window.requestAnimationFrame(() => {
    animate(fun);
  });
};
  
const epicycles = (x, y, rotation, fourier) => {
  for (let i = 0; i < fourier.length; i++) {
    const prevX = x;
    const prevY = y;

    const freq = fourier[i][3];
    const radius = fourier[i][2];
    const loc =  new Complex(fourier[i][0],fourier[i][1]);
    const phase = loc.phase() 

    x += radius * Math.cos(freq * time + phase + rotation);
    y += radius * Math.sin(freq * time + phase + rotation);

    drawLine(prevX, prevY, x, y, circleStroke * 0.2);
    if (showCircle) {
      drawCircle(prevX, prevY, radius, circleStroke);
    }
    if (i === fourier.length - 1) drawCircle(x, y, 2, true);
  }
  return { x, y };
};


const startDrawing = () => {
  const points = epicycles(canvas.width / 2, canvas.height / 2, 0, me);
  
  time+=0.01
  if (time < 2 * Math.PI) {
    graphPoints.unshift(points);
  };
  drawCurve(graphPoints);
  if (graphPoints.length > 10000) graphPoints.pop();
};

//const draw = () => {
//  running = !running;
//  document.querySelector(".start-btn").innerText = running ? "stop" : "start";
//  if (running) {
//    animate(() => {
//      startDrawing();
//    });
//    return;
//  }
//};
animate(() => {
  startDrawing();
})

