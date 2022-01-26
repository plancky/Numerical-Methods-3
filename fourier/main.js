import { me,me_v2 ,me_v3} from "./dat.js";

const canvas = document.querySelector("#canvas");
const dimension = canvas.getBoundingClientRect();
const ctx = canvas.getContext("2d");
let graphPoints = [];
let circleStroke = 1.1;
let lineStroke = 1.3;
let fps = 40;
let timestep = 0.001;
let time = 0;

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

//const animate = () => {setInterval(startDrawing,3);};
//const animate = (fun) => {
//  for (let i = 0; i < speed; i++) {
//    clearCanvas();
//    fun();
//  }
//  window.requestAnimationFrame(() => {
//    animate(fun);
//  });
//};
  
const epicycles = (x, y, rotation, fourier,mode=1) => {
  for (let i = 0; i < fourier.length; i++) {
    const prevX = x;
    const prevY = y;

    const freq = fourier[i][3];
    const radius = fourier[i][2];
    const loc =  new Complex(fourier[i][0],fourier[i][1]);
    const phase = loc.phase() 

    x += radius * Math.cos(2 * Math.PI* freq * time + phase + rotation);
    y += radius * Math.sin(2 * Math.PI* freq * time + phase + rotation);

    drawLine(prevX, prevY, x, y, circleStroke * 0.2);
    drawCircle(prevX, prevY, radius, circleStroke);
    if (i === fourier.length - 1) drawCircle(x, y, 2, true);
  
  }
  return { x, y };
};


function startDrawing(coeff,x= canvas.width/2 ,y= canvas.height /2,mode=0,dt = 0.001) {
  if (time <= 1.2) {
    const points = epicycles(x, y, 0, coeff,mode);
    graphPoints.unshift(points);
  }
  else{
    if (mode==1) epicycles(x, y, 0, coeff,mode);
  }
  drawCurve(graphPoints);
  if (graphPoints.length > 10000) graphPoints.pop();
  time+=dt;
}

const setup_new_frame = () => {
  clearCanvas();
  startDrawing(me_v3,canvas.width/2,canvas.height/2,1);

};

//animate(() => {
//  startDrawing();
//})

//animate();

setInterval(setup_new_frame,1000/fps);