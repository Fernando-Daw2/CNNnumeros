let model = null;

const canvas=document.getElementById("canvas");
const ctx=canvas.getContext("2d");

ctx.fillStyle="black";
ctx.fillRect(0,0,280,280);

ctx.strokeStyle="white";
ctx.lineWidth=18;
ctx.lineCap="round";

let drawing=false;



canvas.onmousedown=()=>drawing=true;
canvas.onmouseup=()=>{

drawing=false;

ctx.beginPath();

};

canvas.onmousemove=(e)=>{

if(!drawing)return;

const rect=canvas.getBoundingClientRect();

const x=e.clientX-rect.left;
const y=e.clientY-rect.top;

ctx.lineTo(x,y);

ctx.stroke();

ctx.beginPath();

ctx.moveTo(x,y);

};


async function loadModel() {

    try{

        model = await tf.loadGraphModel("modelo_web/model.json");

        console.log("Modelo cargado");

        console.log(model);

        console.log("Entradas:", model.inputs);

        console.log("Salidas:", model.outputs);

    }catch(e){

        console.error(e);

    }

}

loadModel();


document.getElementById("clearBtn").onclick=()=>{

ctx.fillStyle="black";

ctx.fillRect(0,0,280,280);

};

document.getElementById("predictBtn").onclick=predict;

async function predict(){
if(model==null){

    alert("El modelo todavía no ha terminado de cargarse.");

    return;

}
const t0=performance.now();

const small=document.createElement("canvas");

small.width=28;

small.height=28;

const sctx=small.getContext("2d");

sctx.drawImage(canvas,0,0,28,28);

const preview=document.getElementById("preview");

const pctx=preview.getContext("2d");

pctx.imageSmoothingEnabled=false;

pctx.clearRect(0,0,140,140);

pctx.drawImage(small,0,0,140,140);

const img=sctx.getImageData(0,0,28,28);

let pixels=[];

for(let i=0;i<img.data.length;i+=4){

pixels.push(img.data[i]/255);

}

const input=tf.tensor4d(pixels,[1,28,28,1]);

const output = await model.executeAsync({
    "keras_tensor": input
});

let probs;

if (Array.isArray(output)) {
    probs = await output[0].data();
    output[0].dispose();
} else {
    probs = await output.data();
    output.dispose();
}

input.dispose();


let max=0;

let digit=0;

for(let i=0;i<10;i++){

if(probs[i]>max){

max=probs[i];

digit=i;

}

}

const t1=performance.now();

document.getElementById("prediction").innerHTML="Predicción: "+digit;

document.getElementById("confidence").innerHTML="Confianza: "+(max*100).toFixed(2)+"%";

document.getElementById("time").innerHTML="Tiempo: "+(t1-t0).toFixed(2)+" ms";

drawBars(probs);

}

function drawBars(probs){

const div=document.getElementById("bars");

div.innerHTML="";

for(let i=0;i<10;i++){

const row=document.createElement("div");

row.className="bar";

const label=document.createElement("div");

label.className="label";

label.innerHTML=i;

const fill=document.createElement("div");

fill.className="fill";

fill.style.width=(probs[i]*400)+"px";

const value=document.createElement("div");

value.className="value";

value.innerHTML=(probs[i]*100).toFixed(2)+"%";

row.appendChild(label);

row.appendChild(fill);

row.appendChild(value);

div.appendChild(row);

}

}