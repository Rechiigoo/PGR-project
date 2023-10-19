#ifdef VERTEX_SHADER
#line 144
precision highp float;

uniform mat4 view;
uniform mat4 proj;

layout(std430,binding=0)buffer SHBuffer{float shBuffer[];};
uniform uint lineLength = 42;
uniform uint nofSH = 1u;
uniform float scaleFactor = 0.01;
uniform float normalLen = 10;
uniform int m_bands = 1;

#define SH_NORMTBL_SIZE 10

int factorial(int a){
  int r=1;
  for(int i=2;i<=a;++i)
    r*=i;
  return r;
}


float computeNormalization(int l, int m) {
	return sqrt(((2*l+1) * factorial(l-m)) / (4 * radians(180) * factorial(l+m)));
}


float legendreP(int l, int m, float x) {
	/* Evaluate the recurrence in double precision */
	double p_mm = 1;

	if (m > 0) {
		double somx2 = sqrt((1 - x) * (1 + x));
		double fact = 1;
		for (int i=1; i<=m; i++) {
			p_mm *= (-fact) * somx2;
			fact += 2;
		}
	}

	if (l == m)
		return float(p_mm);

	double p_mmp1 = x * (2*m + 1) * p_mm;
	if (l == m+1)
		return float(p_mmp1);

	double p_ll = 0;
	for (int ll=m+2; ll <= l; ++ll) {
		p_ll = ((2*ll-1)*x*p_mmp1 - (ll+m-1) * p_mm) / (ll-m);
		p_mm = p_mmp1;
		p_mmp1 = p_ll;
	}

	return float(p_ll);
}

float coef(uint sh,int l,int m){
  return shBuffer[lineLength*sh+12u+uint(l*(l+1) + m)];
}

float eval(uint sh,float theta,float phi){
  float result = 0;
  float cosTheta = cos(theta);

  for (int l = 0; l < m_bands; ++l) {
      for (int m = 1; m <= l; ++m) {
          float L = legendreP(l, m, 2.0f * cosTheta - 1) * computeNormalization(l, m);
          result += coef(sh,l, -m) * sqrt(2) * sin((m-1+1)*phi) * L;
          result += coef(sh,l, +m) * sqrt(2) * cos((m-1+1)*phi) * L;
      }

      result += coef(sh,l, 0) * legendreP(l, 0, 2.0f * cosTheta - 1) * computeNormalization(l, 0);
  }
  return result;
}

out vec3 vColor;
out vec2 vTexCoord;
out vec3 vNormal;

uint hemisphere(uint vID,uint shID,in uint offset,in mat4 vp,in mat4 model,in uint nCols,in uint nRows,in vec3 center,in float radius){
  uint nCells = nCols * nRows;
  uint verticesPerCell = 6u;
  uint nVertices = nCells * verticesPerCell;
  uint verticesPerRow = nCols * verticesPerCell;

  //uint vID = uint(gl_VertexID);
  //if(vID < offset || vID >= offset+nVertices)return nVertices;

  uint cellID = vID / verticesPerCell;
  uint verID  = vID % verticesPerCell;

  uint rowID = cellID / nCols;
  uint colID = cellID % nCols;

  uvec2 verOffset[] = uvec2[](
    uvec2(0,0),
    uvec2(1,0),
    uvec2(0,1),
    uvec2(0,1),
    uvec2(1,0),
    uvec2(1,1)
  );

  float zAngleNorm = float(colID + verOffset[verID].x * 1.0f) / float(nCols);
  float yAngleNorm = float(rowID + verOffset[verID].y * 1.0f) / float(nRows);


  float yAngle = radians(yAngleNorm * 90.f);
  float zAngle = radians(zAngleNorm * 360.f);

  vec2 xyCircle = vec2(         cos(zAngle),sin(zAngle));
  vec3 sphere   = vec3(xyCircle*cos(yAngle),sin(yAngle));

  vTexCoord = vec2(zAngleNorm,yAngleNorm);
  vNormal   = normalize(vec3(inverse(transpose(model))*vec4(sphere,0)));
  float shR = eval(shID,yAngle,zAngle);
  gl_Position = vp * model * vec4(sphere*radius*shR+center,1.f);

  return nVertices;
}


uniform uint nC = 20;
uniform uint nR = 20;
uniform float shScale = 10.0;

void main(){
  gl_Position = vec4(0.f,0.f,0.f,1.f);

  uint nofVerticesPerSH = nC*nR*6u;

  if(gl_VertexID>=nofVerticesPerSH*nofSH)return;

  uint shID = uint(gl_VertexID/nofVerticesPerSH);
  uint vID = uint(gl_VertexID)%nofVerticesPerSH;

  vec3 point;
  vec3 normal;
  point[0] = shBuffer[shID*lineLength+0];
  point[1] = shBuffer[shID*lineLength+1];
  point[2] = shBuffer[shID*lineLength+2];
  normal[0] = shBuffer[shID*lineLength+9+0];
  normal[1] = shBuffer[shID*lineLength+9+1];
  normal[2] = shBuffer[shID*lineLength+9+2];

  normal = normalize(normal);

  vec4 pp;
  mat4 sceneScale = mat4(1);
  sceneScale[0][0] = sceneScale[1][1] = sceneScale[2][2] = scaleFactor;

  
  vec3 t,b;

  t[0] = shBuffer[shID*lineLength+3+0];
  t[1] = shBuffer[shID*lineLength+3+1];
  t[2] = shBuffer[shID*lineLength+3+2];
  t=normalize(t);
  //b = normalize(cross(normal,t));
  b[0] = shBuffer[shID*lineLength+6+0];
  b[1] = shBuffer[shID*lineLength+6+1];
  b[2] = shBuffer[shID*lineLength+6+2];
  b=normalize(b);
  
  //t = normalize(cross(normal,cross(normal,vec3(1,1,1))));
  //b = normalize(cross(normal,t));


  mat4 tbn = mat4(1);
  tbn[0].xyz = t;
  tbn[1].xyz = b;
  tbn[2].xyz = normal;

  mat4 translate = mat4(1);
  translate[3].xyz = point;

  mat4 vp = proj*view;
  mat4 model = sceneScale*translate*tbn;


  uint offset = 0u;
  offset += hemisphere(vID,shID,offset,vp,model,nC,nR,vec3(0,0,0),shScale);


/*
  uint shID = uint(gl_VertexID/2);
  uint pointID = gl_VertexID%2;

  vec3 point;
  vec3 normal;
  point[0] = shBuffer[shID*lineLength+0];
  point[1] = shBuffer[shID*lineLength+1];
  point[2] = shBuffer[shID*lineLength+2];
  normal[0] = shBuffer[shID*lineLength+3+0];
  normal[1] = shBuffer[shID*lineLength+3+1];
  normal[2] = shBuffer[shID*lineLength+3+2];

  vec4 pp;
  if(pointID == 0){
    pp = vec4(point,1);
    vColor = vec3(0);
  }else{
    pp = vec4(point+normal*normalLen,1); 
    vColor = vec3(1);
  }

  mat4 model = mat4(1);
  model[0][0] = model[1][1] = model[2][2] = scaleFactor;

  gl_Position = proj*view*model*pp;
*/
}
#endif



#ifdef FRAGMENT_SHADER

layout(location=0)out vec4 fColor;

in vec3 vNormal;

void main(){
  fColor = vec4(vNormal,1);
}
#endif
