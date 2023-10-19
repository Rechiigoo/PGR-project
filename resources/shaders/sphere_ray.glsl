#define JOIN1(x,y) x##y
#define JOIN0(x,y) JOIN1(x,y)
#define JOIN(x,y)  JOIN0(x,y)

#define VEC1 float
#define VEC2 vec2
#define VEC3 vec3
#define VEC4 vec4

#define IVEC1 int
#define IVEC2 ivec2
#define IVEC3 ivec3
#define IVEC4 ivec4

#define UVEC1 uint
#define UVEC2 uvec2
#define UVEC3 uvec3
#define UVEC4 uvec4

#define VECI1(x,i) x
#define VECI2(x,i) x[i]
#define VECI3(x,i) x[i]
#define VECI4(x,i) x[i]
#define VECXI(x,m,i) JOIN(VECI,m)(x,i)

const uint UINT_NULA                   = 0u;
const uint UINT_JEDNA                  = 1u;
const uint UINT_DVA                    = 2u;
const uint UINT_DESET                  = 10u;
const uint UINT_DVACETTISICDVACETCTYRI = 20024u;
const uint UINT_MAXDELENODVEMA         = 0x7fffffffu;
const uint UINT_MAX                    = 0xffffffffu;

uint poly(in uint x,in uint c){
  //return x*(x*(x+c)+c);
  return x*(x*(x*(x*(x+c)+c)+c)+c);
}

#define BASE(DIM) \
float bnoise(in JOIN(UVEC,DIM) x){\
  uint last=UINT_DESET;\
  for(uint i=UINT_NULA;i<uint(DIM);++i)\
    last=poly(VECXI(x,DIM,i)+(UINT_DVACETTISICDVACETCTYRI<<i),last);\
  return -1.+float(last)/float(UINT_MAXDELENODVEMA);\
}


#define SMOOTH(DIM)\
float snoise(in uint d,in JOIN(UVEC,DIM) x){\
  if(d==UINT_NULA)return bnoise(x);\
  uint dd=UINT_JEDNA<<d;\
  uint mm=(UINT_MAX>>d)+UINT_JEDNA;\
  JOIN(UVEC,DIM) xx=x>>d;\
  JOIN(VEC,DIM) t=JOIN(VEC,DIM)(x%dd)/JOIN(VEC,DIM)(dd);\
  float ret=0.;\
  for(uint i=UINT_NULA;i<(UINT_JEDNA<<DIM);++i){\
    float coef=1.;\
    JOIN(UVEC,DIM) o;\
    for(uint j=UINT_NULA;j<uint(DIM);++j){\
      VECXI(o,DIM,j)=(i>>j)&UINT_JEDNA;\
      coef*=smoothstep(0,1,float(UINT_JEDNA-((i>>j)&UINT_JEDNA))*(1.-2.*VECXI(t,DIM,j))+VECXI(t,DIM,j));\
    }\
    ret+=bnoise((xx+o)%mm)*(coef);\
  }\
  return ret;\
}

#define OCTAVE(DIM)\
float noise(in JOIN(UVEC,DIM) x,in uint M,in uint N,float p){\
  float ret=0.;\
  float sum=0.;\
  for(uint k=UINT_NULA;k<=N;++k){\
    sum*=p;\
    sum+=1;\
    ret*=p;\
    ret+=snoise(M-k,x);\
  }\
  return ret/sum;\
}

#define OCTAVE_SIMPLIFIED(DIM)\
float noise(in JOIN(UVEC,DIM) x,in uint M){\
  return noise(x,M,M,2.);\
}

#define FNOISE(DIM)\
float noise(in JOIN(VEC,DIM) x,in uint M){\
  return noise(JOIN(UVEC,DIM)(JOIN(IVEC,DIM)(x)),M);\
}

#define INOISE(DIM)\
float noise(in JOIN(IVEC,DIM) x,in uint M,in uint N,float p){\
  return noise(JOIN(UVEC,DIM)(x),M,N,p);\
}

BASE(1)
BASE(2)
BASE(3)
BASE(4)

SMOOTH(1)
SMOOTH(2)
SMOOTH(3)
SMOOTH(4)

OCTAVE(1)
OCTAVE(2)
OCTAVE(3)
OCTAVE(4)

OCTAVE_SIMPLIFIED(1)
OCTAVE_SIMPLIFIED(2)
OCTAVE_SIMPLIFIED(3)
OCTAVE_SIMPLIFIED(4)

FNOISE(1)
FNOISE(2)
FNOISE(3)
FNOISE(4)

INOISE(1)
INOISE(2)
INOISE(3)
INOISE(4)

#define BEGINGRADIENT(name)\
vec4 name(float t){        \
  t=clamp(t,0.f,1.f);      \
  const vec4 colors[]={

#define ENDGRADIENT                        \
  };                                       \
  t *= colors.length();                    \
  uint i = uint(floor(t));                 \
  uint j = i + 1u;                         \
  if(j >= colors.length()){                \
    i = colors.length() - 1u;              \
    j = i;                                 \
  }                                        \
  return mix(colors[i],colors[j],fract(t));\
}

#ifdef VERTEX_SHADER
#line 2
precision highp float;

out vec2 vPos;

void main(){
  gl_Position = vec4(0.f,0.f,0.f,1.f);

  if(gl_VertexID>=6)return;
  vPos = vec2((0x32>>gl_VertexID)&1,(0x2c>>gl_VertexID)&1)*2.f-1.f;
  gl_Position = vec4(vPos,0,1);
}
#endif



#ifdef FRAGMENT_SHADER

vec3 random3D[] = vec3[](
vec3(0.41974257546031934, 0.43335567323165075, 0.7975080130154685),
vec3(0.8285995578880808, 0.5591840646009707, 0.0271284825232487),
vec3(0.5035765808737189, 0.8267330314156152, 0.25084481649394785),
vec3(0.8323473706233658, 0.5433603951255489, 0.10934960276734776),
vec3(0.860908939424206, 0.34618573410009423, 0.3728152833845033),
vec3(0.810389507716465, 0.573239292185591, 0.12110144374708401),
vec3(0.7667180316265997, 0.6326226883823898, 0.10923366753190215),
vec3(0.2975436079816183, 0.1465480666660106, 0.9433935899219029),
vec3(0.7858621069379166, 0.6181164267717333, 0.01878381840955004),
vec3(0.3039667138261821, 0.5798502382863605, 0.7558954544412627),
vec3(0.8175588389535181, 0.28591655063643573, 0.4998492481950313),
vec3(0.5728479801485852, 0.546816349793657, 0.6106038578636949),
vec3(0.4217430837339423, 0.602354001501792, 0.6777185464463579),
vec3(0.5618852012505587, 0.5227958006821122, 0.6410690847364022),
vec3(0.7465059696501174, 0.6653523292225071, 0.0059258142834853296),
vec3(0.807551884520158, 0.44432475001414545, 0.38786011696590894),
vec3(0.6617998383122398, 0.008627954851427767, 0.7496309307952649),
vec3(0.26956478350708185, 0.8864489538676699, 0.37622211482008044),
vec3(0.546823223352372, 0.7989669244533164, 0.25027228378733973),
vec3(0.4153900126088934, 0.8621260097351496, 0.29015492545006394),
vec3(0.39542783790282376, 0.10872108007973412, 0.9120397753156352),
vec3(0.02086379799961319, 0.6411801042641841, 0.7671067564744827),
vec3(0.40871291554890427, 0.7201458447865963, 0.5606636379328642),
vec3(0.10416300182734341, 0.9821678446556777, 0.1565132389759322),
vec3(0.9815110323571711, 0.0832484636432283, 0.1723536673883164),
vec3(0.06980527750906636, 0.6020871723062737, 0.7953730320900487),
vec3(0.06458452321195685, 0.9841514808957098, 0.1651505434815934),
vec3(0.25020328937710834, 0.7152957313521296, 0.6524954641177952),
vec3(0.8907073405091452, 0.3677195955294996, 0.26725031829119805),
vec3(0.18229432114405353, 0.6303580214472038, 0.7545976048701687),
vec3(0.7272822209998145, 0.2639259256903204, 0.6335642641169754),
vec3(0.43218088275501587, 0.3230449340465933, 0.8419392229656054),
vec3(0.4443185864210428, 0.45588877020227847, 0.7711980439318167),
vec3(0.6570488511853113, 0.5985455303349836, 0.4582903613126464),
vec3(0.7118312156957574, 0.19783299733394394, 0.673912772936504),
vec3(0.7702061537163569, 0.3945268305423927, 0.5011297843469592),
vec3(0.22823213661962327, 0.599206616664365, 0.7673731311165946),
vec3(0.8118280225440883, 0.5628609232848757, 0.15531530140669508),
vec3(0.39510437722627967, 0.9087904067075616, 0.13413622841327522),
vec3(0.857962574719816, 0.4950321597878115, 0.13727119565283427),
vec3(0.2346392633974271, 0.7551237883602424, 0.6121539678257332),
vec3(0.24105879156061077, 0.885055639178435, 0.3982049404135891),
vec3(0.9413151203882544, 0.18732857719538237, 0.28077366025751044),
vec3(0.7381062841734927, 0.27152316419054157, 0.6176360453953084),
vec3(0.8345434816284452, 0.0899707635831403, 0.5435461700460595),
vec3(0.6143779978896883, 0.7661418652286841, 0.1885903445380248),
vec3(0.7630161382421664, 0.19265438587544323, 0.6170013455293004),
vec3(0.5628321779736332, 0.5734531553785015, 0.5952910364047966),
vec3(0.701716508312513, 0.5227315305087223, 0.4840926450316135),
vec3(0.6351525555933569, 0.6804535156625874, 0.3654644225443194),
vec3(0.8927603484955394, 0.3797773052309395, 0.24238019429328145),
vec3(0.7582691545987871, 0.013604452167456033, 0.6517996686599835),
vec3(0.7280586790015023, 0.05151057154528624, 0.6835767849698124),
vec3(0.29131255812338586, 0.619694878058576, 0.7287765443450932),
vec3(0.13954392165344306, 0.7260967775446686, 0.67328371699368),
vec3(0.7341866481529445, 0.08483653177696132, 0.6736265497662641),
vec3(0.43164022234745925, 0.7439330236200487, 0.5101474049913103),
vec3(0.42508444244717736, 0.4924936843359394, 0.7594426822865467),
vec3(0.4524798196497801, 0.6462040707219451, 0.6145586316960245),
vec3(0.32039955100981815, 0.4317112790463534, 0.8431900730303155),
vec3(0.061764327389068466, 0.22929830027257883, 0.9713945940524285),
vec3(0.27796874810416283, 0.026669757473757914, 0.9602198181217129),
vec3(0.6754723555909674, 0.673703839853254, 0.29976696448970286),
vec3(0.798780206864841, 0.4713011093068689, 0.37393240764485314),
vec3(0.783429901409577, 0.010194152409336429, 0.6213965471693866),
vec3(0.695650082942793, 0.6860129408014369, 0.21320695850428859),
vec3(0.5828592876496, 0.047911478478803594, 0.811159380781962),
vec3(0.09674863034872308, 0.8455058449766003, 0.5251281449665895),
vec3(0.7273233826334407, 0.6822867449743989, 0.07406412564048404),
vec3(0.4656761192349422, 0.6212032512245659, 0.6302795194533238),
vec3(0.6937329157265901, 0.03644723992002166, 0.7193095580761429),
vec3(0.31029499753878775, 0.37497189508061024, 0.8735634449781308),
vec3(0.40302844151014255, 0.8074296502036623, 0.4308427036702575),
vec3(0.09890817943842202, 0.9231724885558205, 0.37144276600013454),
vec3(0.12721949233443686, 0.8502805317641201, 0.5107232304028214),
vec3(0.5657017216432562, 0.0324755963384529, 0.8239702044202313),
vec3(0.1644582642805023, 0.15849169835734606, 0.9735675943978664),
vec3(0.8442219919732944, 0.3247812570301151, 0.42638757410433664),
vec3(0.052534460617278, 0.76127769903266, 0.6462943566233503),
vec3(0.33065036250915664, 0.8950892058969726, 0.29914152379649717),
vec3(0.5182944744855197, 0.2402507679021452, 0.8207620886957491),
vec3(0.23032042735663252, 0.9516667428308713, 0.20318196605983121),
vec3(0.8915561530937511, 0.3563422199609339, 0.2795493662199667),
vec3(0.5589866584033071, 0.28309264698465336, 0.779353879152678),
vec3(0.2361705241798521, 0.1856212522153656, 0.9538177154124438),
vec3(0.7513839239309275, 0.478316077897667, 0.45457224781409056),
vec3(0.607060317246445, 0.08136769554311378, 0.7904790125908774),
vec3(0.5939859222106237, 0.011858639808008537, 0.8043880263141034),
vec3(0.05376170628827007, 0.7726851090842471, 0.6325088150661923),
vec3(0.043051688421084866, 0.8121310749380647, 0.5818845841264723),
vec3(0.48871700545601465, 0.28177110547362755, 0.8256880359423138),
vec3(0.9512733745159735, 0.30394314585008025, 0.05193776109681152),
vec3(0.9760401256321304, 0.12763082955118923, 0.17622725244436638),
vec3(0.41687093370279404, 0.816373646642189, 0.39969074758104556),
vec3(0.6282053313400725, 0.7769882866066086, 0.0405865020916616),
vec3(0.6189896453582773, 0.7651880916757234, 0.1770282556455299),
vec3(0.7592640463997457, 0.47483551419837927, 0.44502735005912364),
vec3(0.8359425127905497, 0.05083431284413084, 0.5464576726033653),
vec3(0.5165392089608555, 0.8562589583410083, 0.002800333348927105),
vec3(0.5467608238573122, 0.665274925196427, 0.5083914588186567),
vec3(0.12267491521682125, 0.7443259742255864, 0.6564523663371755),
vec3(0.3568792557137951, 0.8350809393991446, 0.41866098635221544),
vec3(0.4809866687655827, 0.05202938901189033, 0.8751827050101235),
vec3(0.7434413175805556, 0.19267524140865064, 0.6404461403289161),
vec3(0.4853194725755857, 0.53740317225332, 0.6896831446331162),
vec3(0.6876492518770034, 0.5216286899544575, 0.5050168474510469),
vec3(0.882821816833384, 0.4558573860735543, 0.11322404022639937),
vec3(0.9741098666961645, 0.22490905783188264, 0.022932145786113155),
vec3(0.5954456380436299, 0.5009110408388315, 0.6281183179151623),
vec3(0.5600560968108877, 0.6243939702465258, 0.5444899800223456),
vec3(0.4386841926765939, 0.7648968833889296, 0.47168732957075216),
vec3(0.9839838542444049, 0.05360591796435661, 0.17000641207179304),
vec3(0.8569905111899826, 0.5143968624672897, 0.0310343618291161),
vec3(0.5350828958894797, 0.815924585171395, 0.2189825697159825),
vec3(0.03146842051975181, 0.783779608951902, 0.6202412942566754),
vec3(0.8210537684222255, 0.3556059483273644, 0.44655919974159997),
vec3(0.6226458646170449, 0.6237767738836846, 0.47245599121881726),
vec3(0.12689435170347646, 0.6280910383205287, 0.7677235642385836),
vec3(0.8886690991740627, 0.3404116778245064, 0.30722487167197976),
vec3(0.4198404503362595, 0.49508844899572607, 0.7606716926062473),
vec3(0.7944483130407687, 0.40399381169799786, 0.45347643601007465),
vec3(0.011824317630694591, 0.8598459119250863, 0.5104166859128776),
vec3(0.5492837048893259, 0.6689605851669964, 0.5007785409101367),
vec3(0.41605205358279673, 0.27954278903211394, 0.8653071811846267),
vec3(0.05060786583777024, 0.7366471397776112, 0.674381075781943),
vec3(0.6516995988852441, 0.15173869688829703, 0.7431439972706861),
vec3(0.5654734270404174, 0.8122025640006405, 0.14341129084542562),
vec3(0.7330064148991042, 0.6604984923205387, 0.16261407491068378),
vec3(0.44915609478918256, 0.6855083951712484, 0.5730070179879742),
vec3(0.6071318838979857, 0.7912341358487764, 0.07307132010661091),
vec3(0.9748501677714457, 0.22015790869494029, 0.03460701713604417),
vec3(0.7107651511863429, 0.548852853668145, 0.43996982269172624),
vec3(0.5159750287565652, 0.293362196764699, 0.8048033245514409),
vec3(0.3765536357262917, 0.3337628337209212, 0.8641815377846787),
vec3(0.4598175792860424, 0.5380957307194256, 0.7064140275794722),
vec3(0.329921651861467, 0.8536448781504549, 0.40304109671408916),
vec3(0.06875598837523042, 0.8889754105737002, 0.45276410354385027),
vec3(0.18583374484176404, 0.953612142793311, 0.23683264216580432),
vec3(0.7077483030674732, 0.5353200280274913, 0.46100412915478106),
vec3(0.8668300388555894, 0.2563611779593696, 0.4276501258890285),
vec3(0.3893369395771023, 0.4507593429945864, 0.8032638185452055),
vec3(0.31670076824724425, 0.3445689867318155, 0.8837266753777524),
vec3(0.3808959083951742, 0.561660071218306, 0.7344768691843621),
vec3(0.7451013059939375, 0.5850610691444124, 0.32019929602940406),
vec3(0.5891977607441469, 0.6843329536698677, 0.42957468181395186),
vec3(0.4027881075520754, 0.7938409264481726, 0.45560764250671293),
vec3(0.6318942565910141, 0.3195202992687714, 0.7061277694882736),
vec3(0.6291312156359253, 0.02268812021823008, 0.776967929012148),
vec3(0.8383001555236074, 0.18003022821359602, 0.5146279881413915),
vec3(0.7964809057373621, 0.4609442174841445, 0.39134204369543685),
vec3(0.49452595894499307, 0.5834366503191318, 0.6442404450156215),
vec3(0.6857137508460814, 0.6520154882730702, 0.3235312271676851),
vec3(0.7977717655561036, 0.26915936895232806, 0.5395492972719749),
vec3(0.760774669074316, 0.48777385836957327, 0.4281338178491913),
vec3(0.30723206143340337, 0.18760856416052507, 0.9329584594616244),
vec3(0.8232328688781629, 0.4777793108881462, 0.30661796047504725),
vec3(0.17654489444748883, 0.5268699587044379, 0.8314084115880166),
vec3(0.9310725824932784, 0.3468118365815921, 0.11324926549960627),
vec3(0.05072167367080172, 0.9824146509355993, 0.179690749475641),
vec3(0.7592708880017242, 0.12010106725214262, 0.6395963197814459),
vec3(0.25554805280728804, 0.7127249420339008, 0.6532368251324906),
vec3(0.8165996465316129, 0.14515685877705278, 0.5586541896687284),
vec3(0.8695917944619388, 0.20215018467067025, 0.450494632423161),
vec3(0.7764668144445295, 0.35808098571918595, 0.5185337922765889),
vec3(0.41418302327818995, 0.9033677910417423, 0.1112612121833187),
vec3(0.4759521292204175, 0.4905978760272149, 0.7299200605053519),
vec3(0.7396252975232005, 0.6481434285222955, 0.18128572841534243),
vec3(0.3249175151192699, 0.8457668402640873, 0.42321042080556204),
vec3(0.6680803427086613, 0.34448188502423943, 0.6595459700251561),
vec3(0.5870669704802949, 0.7609175053436609, 0.2763275632879724),
vec3(0.7326602495453053, 0.5614113490965323, 0.38474180412560116),
vec3(0.20462359098923832, 0.04877367986225477, 0.9776248330332875),
vec3(0.4027645327939703, 0.748674913404196, 0.5265611124669902),
vec3(0.06837773468944913, 0.9976369259099622, 0.006711889425477347),
vec3(0.5577245187899372, 0.24723697724260138, 0.7923491895777228),
vec3(0.8450802441743391, 0.37726077018942356, 0.37883201050376214),
vec3(0.996263941793572, 0.03998756285163558, 0.07654510499777542),
vec3(0.6613762977995776, 0.13058360094721688, 0.7385995639536909),
vec3(0.7570274673215007, 0.602003349474887, 0.2539712206959898),
vec3(0.9603167075993267, 0.13072157714636454, 0.24638118916418142),
vec3(0.4458596374285981, 0.17416176922023047, 0.8779959349872403),
vec3(0.9907775025098999, 0.012680246781354067, 0.13490423218642655),
vec3(0.8562176034052387, 0.20917960160806714, 0.47237200371114274),
vec3(0.551748222487684, 0.0951609980665704, 0.8285639887351219),
vec3(0.8000981638662313, 0.5996072468536612, 0.017722237399880585)
);

vec3 rand3D(uint a){
  a=a*(a*(a*(a*(a+1321u)+1337u)+199331u)+33231u)+77373u;
  float t = float(a)/float(0xffffffffu);
  uint i = uint(t*random3D.length());
  return normalize(random3D[i]*2.-1.);
}


precision highp float;
precision highp float;

uniform mat4 view;
uniform mat4 proj;
uniform float iTime;

out highp vec4 fColor;

struct Ray{
  vec3 s;
  vec3 d;
  float minT;
  float matT;
};

struct Sphere{
  vec3 c;
  float r;
  vec3 color;
  float metalness;
  float roughness;
  float opticalDensity;
};

float raySphere(Ray ray,Sphere s){
  float a = dot(ray.d,ray.d);
  float b = dot(ray.s-s.c,ray.d)*2.f;
  float c = dot(ray.s-s.c,ray.s-s.c)-s.r*s.r;
  float d = b*b-4.f*a*c;
  float t0 = (-b - sqrt(d))/(2.f*a);
  float t1 = (-b + sqrt(d))/(2.f*a);
  if(t0 < 0.f && t1 < 0.f)return -1.f;
  if(t0 < 0.f)return t1;
  if(t1 < 0.f)return t0;
  return min(t0,t1);
}

const float size = 100000;
const float roomSize = 1000;

const Sphere scene[]=Sphere[](
   Sphere(vec3(-size,0    ,    0),size-roomSize,vec3(1,0,0),0,1,100)
  ,Sphere(vec3(+size,0    ,    0),size-roomSize,vec3(0,1,0),0,1,100)
  ,Sphere(vec3(    0,0    ,-size),size-roomSize,vec3(.5   ),0,1,100)
  ,Sphere(vec3(    0,-size,    0),size-roomSize,vec3(.5   ),0,1,100)
  ,Sphere(vec3(    0,+size,    0),size-roomSize,vec3(0,0,1),0,1,100)


  ,Sphere(vec3(-700),300,vec3(.3),0,1,1.3)
  ,Sphere(vec3(0  ,0    ,0),10,vec3(0,0,1),0,1,100)
  ,Sphere(vec3(30 ,0    ,0),20,vec3(0,0,1),1,0,100)
  ,Sphere(vec3(60 ,0    ,0),20,vec3(.5,.5,0),.5,.5,100)
  ,Sphere(vec3(300,-300 ,0),200,vec3(0,1,1),1,0,100)

  

);

struct HitPoint{
  float t;
  vec3 position;
  vec3 normal;
  vec3 color;
  float metalness;
  float roughness;
  float opticalDensity;
};

bool visibilityTest(vec3 A,vec3 B){
  Ray ray;
  ray.s = A;
  ray.d = B-A;
  for(int i=0;i<scene.length();++i){
    float t = raySphere(ray,scene[i]);
    if(t < 0.f)continue;
    if(t<1.f)return false;
  }
  return true;
}

HitPoint rayTrace(Ray ray){
  HitPoint hp;
  float t=10e10;
  int id=scene.length();
  for(int i=0;i<scene.length();++i){
    float newt = raySphere(ray,scene[i]);
    if(newt < 0.f)continue;
    if(newt < t){
      id = i;
      t = newt;
    }
  }

  if(id >= scene.length()){
    hp.t = -1.f;
    hp.position = ray.s + ray.d*1000000.f;
    hp.normal   = normalize(-ray.d);
    hp.color    = vec3(0,0,0);
    hp.metalness = 0;
    hp.roughness = 1;
    hp.opticalDensity = 100;
    return hp;
  }

  hp.t = t;
  hp.position  = ray.s + ray.d*t;
  hp.normal    = normalize(hp.position-scene[id].c);
  hp.color     = scene[id].color;

  //if(id == 3)hp.color = vec3(noise(hp.position,8u));//vec3(1,0,1);
  hp.metalness = scene[id].metalness;
  hp.roughness = scene[id].roughness;
  hp.opticalDensity = scene[id].opticalDensity;
  return hp;
}

float BRDF(vec3 n,vec3 o,vec3 i,float roughness,float metalness){
  return max(dot(normalize(i+o),n),0.f);
  float dF = max(dot(n,i),0)*(1-metalness);
  float sF = pow(max(dot(normalize(i+o),n),0),log(roughness))*(1-dF);
  if(roughness==1)return dF;
  if(roughness==0)return float(normalize(i+o)==n);
  return dF + sF;
}

vec3 lightPos = vec3(40+cos(iTime)*100,40,40+sin(iTime)*100);
const vec3 lightColor = vec3(1,1,1);
const vec3 ambientColor = vec3(0.1,0.1,0.1);
vec3 computeDirectIllumination(vec3 outDir,HitPoint hp){
  bool computeDiffuse = visibilityTest(0.99f*hp.position+0.01f*lightPos,lightPos);

  vec3 color = vec3(0);

  vec3 ambientColor = hp.color * ambientColor;
  color += ambientColor;

  if(computeDiffuse){
    vec3 N = hp.normal;
    vec3 L = normalize(lightPos - hp.position);
    float f = BRDF(N,outDir,L,hp.roughness,hp.metalness);
    color += hp.color * lightColor * f;

    //vec3 material = hp.color;
    //float dF = max(dot(N,L),0.f);
    //vec3 diffuseColor = material*dF;
    //color += diffuseColor;
  }

  return color;
}



Ray generateHemisphericRay(vec3 position,vec3 normal,vec3 view,float metalness,float roughness,float opticalDensity,float pix){
  Ray ray;

  if(metalness==1.f){
    ray.d = -reflect(view,normal);
    ray.s = position + ray.d*0.01;
    return ray;
  }
  //if(opticalDensity < 100){
  //  ray.d = refract(view,normal,1/opticalDensity);
  //  ray.s = position
  //  return ray;
  //}
  vec3 vec;
  //vec.x = noise(vec3(1000)+10000*vec3(iTime,pix,  0),8u);
  //vec.y = noise(vec3(1000)+10000*vec3(iTime,pix,100),8u);
  //vec.z = noise(vec3(1000)+10000*vec3(iTime,pix,200),8u);

  vec = rand3D(uint(iTime*512*4+pix));
  vec*=sign(dot(vec,normal)); // flip negative hemisphere


  ray.d = vec;
  ray.s = position+vec*0.03f;
  return ray;
}

vec3 pathTrace(Ray ray,int depth,float pix){
  vec3 finalColor = vec3(0);
  float pathConductibility = 1.f;
  Ray rray = ray;
  for(int d=0;d<depth;++d){
    HitPoint hp = rayTrace(rray);
    vec3 o = -normalize(rray.d);
    finalColor += pathConductibility*computeDirectIllumination(o,hp);
    rray = generateHemisphericRay(hp.position,hp.normal,o,hp.metalness,hp.roughness,hp.opticalDensity,pix);
    pathConductibility *= BRDF(hp.normal,o,rray.d,hp.roughness,hp.metalness);
  }
  return finalColor;
}

in vec2 vPos;

const ivec2 quiltViews = ivec2(5,9);

Ray generateQuiltRay(){
  vec2 vvPos = (vPos*.5+.5)*vec2(quiltViews);
  int viewId = int(vvPos.x)+int(vvPos.y)*5;
  vvPos = fract(vvPos)*2-1;

  mat4 newView = view;
  mat4 newProj = proj;

  
  float aspect = 512.f/512.f;
  float fovy   = 3.1415/2.f;
  float viewCone = 10.f;
  float ttt = float(viewId) / float(45.f - 1);
  float d = 0.7f;
  float S = 0.5f*d*tan(viewCone);
  float s = S-2*ttt*S;

  newView[3][0] += s;
  newProj[2][0] += s/(d*aspect*tan(fovy/2));

  vec4 dir = inverse(newProj*newView)*vec4(vvPos,1,1);
  dir.xyz/=dir.w;
  vec3 pos = vec3(inverse(newProj*newView)*vec4(0,0,0,1));

  Ray ray;
  ray.s=pos;
  ray.d=normalize(dir.xyz);
  return ray;
}

// HoloPlay values
uniform float pitch = 354.677;
uniform float tilt = -0.113949;
uniform float center = -0.400272;
uniform float invView = 1.f;
uniform float flipX;
uniform float flipY;
uniform float subp = 0.000130208;
uniform int ri = 0;
uniform int bi = 2;
uniform vec4 tile = vec4(5,9,45,45);
uniform vec4 viewPortion = vec4(0.99976f, 0.99976f, 0.00f, 0.00f);
uniform vec4 aspect;
uniform float focus = 0.f;


vec2 texArr(vec3 uvz)
{
  float z = floor(uvz.z * tile.z);
  float focusMod = focus*(1-2*clamp(z/tile.z,0,1));
  float x = (mod(z, tile.x) + clamp(uvz.x+focusMod,0,1)) / tile.x;
  float y = (floor(z / tile.x) + uvz.y) / tile.y;
  return vec2(x, y) * viewPortion.xy;
}

uniform  float viewCone = 10.f;

Ray generateChaRay(){
  vec2 texCoords = vPos*.5f+.5f;
	vec3 nuv = vec3(texCoords, 0.0f);

	nuv.z = (texCoords.x + 0 * subp + texCoords.y * tilt) * pitch - center;
	nuv.z = fract(nuv.z);
	nuv.z = (1.0 - nuv.z);
	vec2 vvPos = texCoords*2.f-1.f;//texArr(nuv);

  int viewId = int(nuv.z*tile.z);

  mat4 newView = view;
  mat4 newProj = proj;

  
  float aspect = 512.f/512.f;
  float fovy   = 3.1415/2.f;
  float ttt = float(viewId) / float(45.f - 1);
  float d = 0.7f;
  float S = 0.5f*d*tan(viewCone);
  float s = S-2*ttt*S;

  newView[3][0] += s;
  newProj[2][0] += s/(d*aspect*tan(fovy/2));

  vec4 dir = inverse(newProj*newView)*vec4(vvPos,1,1);
  dir.xyz/=dir.w;
  vec3 pos = vec3(inverse(newProj*newView)*vec4(0,0,0,1));

  Ray ray;
  ray.s=pos;
  ray.d=normalize(dir.xyz);
  return ray;


}

Ray generateFlatScreenRay(){
  vec4 dir = inverse(proj*view)*vec4(vPos,1,1);
  dir.xyz/=dir.w;
  vec3 pos = vec3(inverse(proj*view)*vec4(0,0,0,1));

  Ray ray;
  ray.s=pos;
  ray.d=normalize(dir.xyz);
  return ray;
}

uniform int   drawMode   = 2;
uniform int   maxDepth   = 2;

void main(){
  Ray ray;

  if(drawMode == 0)ray = generateFlatScreenRay();
  if(drawMode == 1)ray = generateQuiltRay     ();
  if(drawMode == 2)ray = generateChaRay       ();

  HitPoint hp = rayTrace(ray);
  if(hp.roughness == 0 && hp.metalness == 1){
    Ray newRay;
    newRay.d = reflect(ray.d,hp.normal);
    newRay.s = hp.position+newRay.d*0.01f;
    hp = rayTrace(newRay);
  }

  vec3 color = computeDirectIllumination(-ray.d,hp);

  color = vec3(0);
  HitPoint prim = rayTrace(ray);
  
  float focusPlane = 18;
  float distanceFromFocusPlane=5;
  int pathsPerPixel;
  pathsPerPixel = 16+int(24*(1-smoothstep(0,1,clamp(abs(length(prim.position - ray.s)-focusPlane)/distanceFromFocusPlane,0,1))));
  if(gl_FragCoord.x<2560/2)pathsPerPixel = 2;
  else pathsPerPixel = 10;
  //pathsPerPixel = 1+64*int(abs(length(prim.position - ray.s)-focusPlane)<distanceFromFocusPlane);
  for(int i=0;i<pathsPerPixel;++i)
    color += pathTrace(ray,maxDepth,iTime*20*gl_FragCoord.x+gl_FragCoord.y*512+i*512*512);
  color /= float(pathsPerPixel);

  fColor = vec4(color,1);

  //fColor = vec4(rand3D(uint(iTime*20*gl_FragCoord.x+gl_FragCoord.y*512)),1);
  //float ttt = iTime;
  //fColor = vec4(sin(ttt),cos(ttt),sin(ttt+3),1);
}
#endif
