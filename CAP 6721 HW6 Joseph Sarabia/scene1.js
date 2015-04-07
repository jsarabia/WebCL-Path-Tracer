"use strict";
function Scene(filename)
{
	var bytesPerFloat = 4;
	
	var nSpheres = 0;
	var nTriangles = 0;
	
	var spheres=[]; // sphere structure: {radius, center, materialIndex}
	var sphereStructureFloats = 1+3+1;
	var triangles=[]; // Triangle structure: {vertex0, vertex1, vertex2, normal0, normal1, normal2, materialIndex}
	var triangleStructureFloats = 6*3+1;
	
	var nMaterials=0;
	var materials=[];
	var materialStructureFloats = 3*3; // Diffuse reflectance and Mirror reflectance
	
	var bounds = {min:[-1,-1,-1], max:[1,1,1]};
	
	var model = parseJSON(filename);
	if (!model) return null;
	parseData();
	//console.log(bounds);
	console.log("sphere data size: "+spheres.length);
	console.log("triangle data size: "+triangles.length);
	console.log("spheres: "+nSpheres+", Triangles: "+nTriangles);
	var sphereBufferSize=nSpheres*sphereStructureFloats*bytesPerFloat;
	var triangleBufferSize=nTriangles*triangleStructureFloats*bytesPerFloat;
	var materialBufferSize=nMaterials*materialStructureFloats*bytesPerFloat;
	this.getNmaterials=function(){ return nMaterials;}
	this.getMaterialBufferSize=function(){ return materialBufferSize;}
	this.getMaterialBufferData=function(){ return new Float32Array(materials);}
	this.getNspheres=function(){ return nSpheres;}
	this.getSphereBufferSize=function(){ return sphereBufferSize;}
	this.getSphereBufferData=function(){ return new Float32Array(spheres);}
	this.getNtriangles=function(){ return nTriangles;}
	this.getTriangleBufferSize=function(){ return triangleBufferSize}
	this.getTriangleBufferData=function(){ return new Float32Array(triangles);}
	this.getBounds=function(){return bounds;}
	this.getViewSpec=function(index){if (model.cameras && model.cameras[index])return model.cameras[index];}
	
	function parseData() 
	{		
		nMaterials = (model.materials)?model.materials.length:0;
		for (var i=0;i<nMaterials;i++){
			var material = model.materials[i];
			materials.push(material.diffuseReflectance[0]);materials.push(material.diffuseReflectance[1]);materials.push(material.diffuseReflectance[2]);
			materials.push(material.specularReflectance[0]);materials.push(material.specularReflectance[1]);materials.push(material.specularReflectance[2]);
			materials.push(material.emissionColor[0]);materials.push(material.emissionColor[1]);materials.push(material.emissionColor[2]);
			
		}
		var xmin, xmax, ymin, ymax, zmin, zmax;
		var firstvertex = true;
		var nNodes = model.nodes.length;
		for (var k=0; k<nNodes; k++){
			var m4 = mat4.clone(model.nodes[k].modelMatrix);
			var m3 = mat3.normalFromMat4(mat3.create(),m4);
			var nMeshes = (model.nodes[k].meshIndices)?model.nodes[k].meshIndices.length:0;
			var nSps = (model.nodes[k].sphereIndices)?model.nodes[k].sphereIndices.length:0;
			var i,j,id,index,n;
			for (n = 0; n < nSps; n++){
				index = model.nodes[k].sphereIndices[n];
				//console.log(nSps+","+index);
				var sphere = model.spheres[index];
				var materialIndex = (sphere.materialIndex)?sphere.materialIndex:0;
				for(i=0;i<sphere.radii.length; i++){
					nSpheres++;
					var center = vec3.transformMat4(vec3.create(),vec3.fromValues(sphere.centers[3*i],sphere.centers[3*i+1],sphere.centers[3*i+2]),m4);
					var rad = sphere.radii[i]; // Assumes that no scaling transformation involved
					spheres.push(rad);
					spheres.push(center[0]);spheres.push(center[1]);spheres.push(center[2]);
					spheres.push(materialIndex);
					updateMinMax([center[0]+rad, center[1]+rad, center[2]+rad]);
					updateMinMax([center[0]-rad, center[1]-rad, center[2]-rad]);
				}
			}
			var vertex, normal;
			for (n = 0; n < nMeshes; n++){
				index = model.nodes[k].meshIndices[n];
				var mesh = model.meshes[index];
				var materialIndex = (mesh.materialIndex)?mesh.materialIndex:0;
				if (mesh.indices){
					console.log("Indices: "+mesh.indices.length);
					for(i=0;i<mesh.indices.length; i+=3){
						nTriangles++;
						for (j = 0; j< 3; j++){
							id = mesh.indices[i+j]*3;
							vertex = vec3.transformMat4(vec3.create(),vec3.fromValues(mesh.vertexPositions[id+0],mesh.vertexPositions[id+1],mesh.vertexPositions[id+2]),m4);
							triangles.push(vertex[0]);triangles.push(vertex[1]);triangles.push(vertex[2]);
							updateMinMax(vertex);
						}
						for (j = 0; j< 3; j++){
							id = mesh.indices[i+j]*3;
							normal = vec3.transformMat3(vec3.create(),vec3.fromValues(mesh.vertexNormals[id+0],mesh.vertexNormals[id+1],mesh.vertexNormals[id+2]),m3);
							triangles.push(normal[0]);triangles.push(normal[1]);triangles.push(normal[2]);
						}
						triangles.push(materialIndex);
					}
				}
				else{
					for(i=0;i<mesh.vertexPositions.length; i+=9){
						nTriangles++;
						for (j = 0; j< 9; j+=3){
							vertex = vec3.transformMat4(vec3.create(),vec3.fromValues(mesh.vertexPositions[i+j],mesh.vertexPositions[i+j+1],mesh.vertexPositions[i+j+2]),m4);
							triangles.push(vertex[0]);triangles.push(vertex[1]);triangles.push(vertex[2]);
							updateMinMax(vertex);
						}
						for (j = 0; j< 9; j+=3){
							normal = vec3.transformMat3(vec3.create(),vec3.fromValues(mesh.vertexNormals[i+j],mesh.vertexNormals[i+j+1],mesh.vertexNormals[i+j+2]),m3);
							triangles.push(normal[0]);triangles.push(normal[1]);triangles.push(normal[2]);
						}
						triangles.push(materialIndex);
					}
				}
			}
		}
		bounds.min = [xmin,ymin,zmin];
		bounds.max = [xmax,ymax,zmax];
		function updateMinMax(vertex)
		{
			if (firstvertex){
				xmin = xmax = vertex[0];
				ymin = ymax = vertex[1];
				zmin = zmax = vertex[2];
				firstvertex = false;
			}
			else{
				if (vertex[0] < xmin) xmin = vertex[0];
				else if (vertex[0] > xmax) xmax = vertex[0];
				if (vertex[1] < ymin) ymin = vertex[1];
				else if (vertex[1] > ymax) ymax = vertex[1];
				if (vertex[2] < zmin) zmin = vertex[2];
				else if (vertex[2] > zmax) zmax = vertex[2];
			}
		}
	}
	function parseJSON(jsonFile)
	{
		console.log(jsonFile);
		var	xhttp = new XMLHttpRequest();
		xhttp.open("GET", jsonFile, false);
		xhttp.overrideMimeType("application/json");
		xhttp.send(null);	
		var Doc = xhttp.responseText;
		return JSON.parse(Doc);
	}
}
