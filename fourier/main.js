import { me,me_v2 ,me_v3,ankur,Harsh} from "./dat.js";

const canvas = document.querySelector("#my_image");
const dimension = canvas.getBoundingClientRect();
const ctx = canvas.getContext("2d");
let graphPoints = [];
let circleStroke = 1.1;
let lineStroke = 1.3;
let fps = 60;
let timestep = 0.001;
let time = 0;


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

class curve {
  constructor (x,id){
    this.coeffs = x;
    this.graphpoints = [];
    this.mode = 1  ;
    this.canvas = document.querySelector(id);
    this.t = 0 ;
    this.id = 0;
    this.stopflag = 0;
    this.canvas.height =  window.innerHeight ;
    this.canvas.width = window.innerWidth;
  }
}

let me_fig = new curve(me_v3,"#my_image");
let ankur_fig = new curve(ankur,"#my_image");
let Harsh_fig = new curve(Harsh,"#my_image");

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

const epicycles = (x, y,time, rotation, fourier,draw=true) => {
  for (let i = 0; i < fourier.length; i++) {
    const prevX = x;
    const prevY = y;

    const freq = fourier[i][3];
    const radius = fourier[i][2];
    const loc =  new Complex(fourier[i][0],fourier[i][1]);
    const phase = loc.phase() 

    x += radius * Math.cos(2 * Math.PI* freq * time + phase + rotation);
    y += radius * Math.sin(2 * Math.PI* freq * time + phase + rotation);
    if (draw){
    drawLine(prevX, prevY, x, y, circleStroke * 0.2);
    drawCircle(prevX, prevY, radius, circleStroke);
    //if (i === fourier.length - 1) drawCircle(x, y, 2, true);
    };
  }
  return { x, y };
};


const startDrawing = (inptcurve,x= canvas.width/2 ,y= canvas.height /2,dt = 0.001) => {
  if (inptcurve.t<=1.1) {
    const points = epicycles(x, y,inptcurve.t, 0, inptcurve.coeffs,!inptcurve.stopflag);
    inptcurve.graphpoints.unshift(points);
  }
  else {
    if (inptcurve.mode == 0){ inptcurve.stopflag = 1; };
  };
  drawCurve(inptcurve.graphpoints);
  inptcurve.t+=dt;
}

const setup_new_frame = () => {
  if (ankur_fig.stopflag && me_fig.stopflag) return;
  clearCanvas();
  startDrawing(ankur_fig,canvas.width/3,canvas.height/2);
  startDrawing(me_fig,canvas.width/3*2,canvas.height/2);
  startDrawing(Harsh_fig,canvas.width/2,canvas.height/4);
  //const id1 = setInterval(setup_new_frame,1000/fps);
  window.requestAnimationFrame(setup_new_frame);
};

ankur_fig.mode =  0;
me_fig.mode =  0;
Harsh_fig.mode = 0;

setup_new_frame();
