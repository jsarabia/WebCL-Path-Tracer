/*
	Joseph Sarabia
	CAP 6721
	Homework 6
*/

var iter = 100;
var sceneLoc = "CornellBoxModel1.json";

function main(){
	 
	var maxint = 2147483647;

	var cl = WebCL.createContext ();
	var device = cl.getInfo(WebCL.CONTEXT_DEVICES)[0];
	var cmdQueue = cl.createCommandQueue (device, 0);
	var programSrc = loadKernel("raytrace");
	var program = cl.createProgram(programSrc);
	try {
		program.build ([device], "");
	} catch(e) {
		alert ("Failed to build WebCL program. Error "
		   + program.getBuildInfo (device, WebCL.PROGRAM_BUILD_STATUS)
		   + ":  " + program.getBuildInfo (device, WebCL.PROGRAM_BUILD_LOG));
		throw e;
	}
	var kernelName = "raytrace";
	try {
		kernel = program.createKernel (kernelName);
	} catch(e){
		alert("No kernel with name:"+ kernelName+" is found.");
		throw e;
	}
	var scene = new Scene(sceneLoc);
	var canvas = document.getElementById("canvas");
	var width=canvas.width, height=canvas.height;
	var canvasContext=canvas.getContext("2d");
	var canvasContent = canvasContext.createImageData(width,height);
	var nPixels = width*height;
	var nChannels = 4;
	var pixelBufferSize = nChannels*nPixels*4;
	var pixelBuffer = cl.createBuffer(WebCL.MEM_READ_WRITE,pixelBufferSize);
	var cameraBufferSize = 40;
	var cameraBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY, cameraBufferSize);
	// [eye,at,up,fov]
	var cameraBufferData = new Float32Array([0,0,1,0,0,0,0,1,0,90]);
	var cameraObj = scene.getViewSpec(0);
	if (cameraObj)
	{
		cameraBufferData[0] = cameraObj.eye[0];
		cameraBufferData[1] = cameraObj.eye[1];
		cameraBufferData[2] = cameraObj.eye[2];
		cameraBufferData[3] = cameraObj.at[0];
		cameraBufferData[4] = cameraObj.at[1];
		cameraBufferData[5] = cameraObj.at[2];
		cameraBufferData[6] = cameraObj.up[0];
		cameraBufferData[7] = cameraObj.up[1];
		cameraBufferData[8] = cameraObj.up[2];
		cameraBufferData[9] = cameraObj.fov;
	}

	var triangleBufferSize = scene.getTriangleBufferSize();
	var triangleBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY,(triangleBufferSize)?triangleBufferSize:1);
	var nTriangles = scene.getNtriangles();

	var sceneData = scene.getTriangleBufferData();

	var nMaterials = scene.getNmaterials();
	var materialBufferSize = scene.getMaterialBufferSize();
	var materialBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY, (materialBufferSize)?materialBufferSize:40);

	var pixelBufferData = new Float32Array(pixelBufferSize);
	for(var i = 0; i<pixelBufferSize; i++)
		pixelBufferData[i] = 0.;

	//light buffer is of form [xmin, zmin, xmax, zmax, y, area, intensityr, intesityg, intensityb]
	var lightBufferData = new Float32Array([-.24, -.22, .23, .26, 1.98, .1316, 17, 12, 4]);
	var lightBufferSize = 36;
	var lightBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY, lightBufferSize);
	canvasBufferSize = nChannels*nPixels;
	canvasBuffer = cl.createBuffer(WebCL.MEM_WRITE_ONLY,canvasBufferSize);
	var randomBuffer = cl.createBuffer(WebCL.MEM_READ_WRITE, nPixels*2*4+1);
	var randomData = [];
	for(var i = 0; i < nPixels*2+1; i++){
		var seed = Math.random() * maxint;		
		if(seed < 2) seed = 2;
		randomData[i] = seed;
	}


	 kernel.setArg(0, pixelBuffer);
	 kernel.setArg(1, cameraBuffer);	
	 kernel.setArg(2, triangleBuffer);
	 kernel.setArg(3, new Int32Array([nTriangles]));
	 kernel.setArg(4, new Int32Array([width]));
	 kernel.setArg(5, new Int32Array([height]));
	 kernel.setArg(6, new Int32Array([nMaterials]));
	 kernel.setArg(7, materialBuffer);
	 kernel.setArg(8, lightBuffer);
	 kernel.setArg(9,randomBuffer);
	 kernel.setArg(10, canvasBuffer);
	 kernel.setArg(11, new Int32Array([iter]));

	var dim = 2;
	var maxWorkElements = kernel.getWorkGroupInfo(device,webCL.KERNEL_WORK_GROUP_SIZE);// WorkElements in ComputeUnit
	var xSize = Math.floor(Math.sqrt(maxWorkElements));
	var ySize = Math.floor(maxWorkElements/xSize);
	var localWS = [xSize, ySize];
	var globalWS = [Math.ceil(width/xSize)*xSize, Math.ceil(height/ySize)*ySize];

	cmdQueue.enqueueWriteBuffer(pixelBuffer, false, 0, pixelBufferSize, new Float32Array(pixelBufferSize));
	cmdQueue.enqueueWriteBuffer(triangleBuffer, false, 0, triangleBufferSize, scene.getTriangleBufferData());
	cmdQueue.enqueueWriteBuffer(materialBuffer, false, 0, materialBufferSize, scene.getMaterialBufferData());
	cmdQueue.enqueueWriteBuffer(cameraBuffer, false, 0, cameraBufferSize, cameraBufferData);
	cmdQueue.enqueueWriteBuffer(lightBuffer, false, 0, lightBufferSize, lightBufferData);
	cmdQueue.enqueueWriteBuffer(randomBuffer, false, 0, nPixels*2*4+1, new Uint32Array(randomData));

	for(var i = 0; i < iter; i++){
		cmdQueue.enqueueNDRangeKernel(kernel,globalWS.length,null,globalWS,localWS);
	}


	cmdQueue.enqueueReadBuffer(canvasBuffer,false,0,canvasBufferSize,canvasContent.data);



	cmdQueue.finish();
	canvasContext.putImageData(canvasContent,0,0);
	pixelBuffer.release();
	cameraBuffer.release();
	cmdQueue.release();
	kernel.release();
	program.release();
	cl.releaseAll();
	cl.release();
}

function loadKernel(id){
  var kernelElement = document.getElementById(id);
  console.log(document.getElementById(id));
  var kernelSource = kernelElement.text;
  if (kernelElement.src != "") {
      var mHttpReq = new XMLHttpRequest();
      mHttpReq.open("GET", kernelElement.src, false);
      mHttpReq.send(null);
      kernelSource = mHttpReq.responseText;
  } 
  return kernelSource;
}

function switchstuff(){
	if(sceneLoc == "CornellBoxModel.json"){
		//type = 1;
		sceneLoc = "CornellBoxModel1.json";
	}
	else {
		sceneLoc = "CornellBoxModel.json";
		//type = 0;
	}
	main();
}

