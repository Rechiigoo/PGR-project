#ifdef VERTEX_SHADER
#line 2
precision highp float;

out vec3 vTexCoord;

uniform uint layers = 4;
uniform float offset = 0.1;

void main(){
  gl_Position = vec4(0.f,0.f,0.f,1.f);

  if(gl_VertexID>=6*layers)return;

  vTexCoord.z = float(gl_VertexID/6)/float(layers);
  vTexCoord.xy = vec2((0x32>>(gl_VertexID%6))&1,(0x2c>>(gl_VertexID%6))&1);
  vTexCoord.x += offset*vTexCoord.z;

  gl_Position = vec4(vTexCoord*2-1,1);
}
#endif



#ifdef FRAGMENT_SHADER

out vec4 fColor;
in vec3 vTexCoord;

layout(binding=0)uniform sampler3D lf_tex;

void main(){

  fColor = vec4(texture(lf_tex,vec3(vTexCoord)).xyz,0.5);

}
#endif
