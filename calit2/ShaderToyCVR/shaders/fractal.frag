
//------------------------------------------------------
//
// Fractal_Vibrations.glsl
//
// original:  https://www.shadertoy.com/view/Xly3R3
//            2016-10-05  Kaleo by BlooD2oo1
//
//   v1.0  2016-10-06  first release
//   v1.1  2018-03-23  AA added, mainVR untested!!! 
//   v1.2  2018-09-02  supersampling corrected
//
// description  a koleidoscopic 3d fractal
//
// Hires B/W fractal picture:
//   https://c2.staticflickr.com/6/5609/15527309729_b2a1d5a491_o.jpg
//
//------------------------------------------------------

float g_fScale = 1.2904082537;

mat4 g_matIterator1 = mat4(-0.6081312299, -0.7035965919, 0.3675977588, 0.0000000000,
                            0.5897225142, -0.0904228687, 0.8025279045, 0.0000000000,
                           -0.5314166546, 0.7048230171, 0.4699158072, 0.0000000000,
                            0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000 );

mat4 g_matIterator2 = mat4(-0.7798885703, 0.6242666245, -0.0454343557, -0.2313748300,
                            0.0581589043, 0.0000002980, -0.9983071089, -0.2313748300,
                           -0.6232098937, -0.7812111378, -0.0363065004, -0.2313748300,
                            0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000 );

mat4 g_matReflect1 = mat4( 0.9998783469, -0.0103046382, -0.0117080826, 0.0000000000,
                          -0.0103046382, 0.1270489097, -0.9918430448, 0.0000000000,
                          -0.0117080826, -0.9918430448, -0.1269274950, 0.0000000000,
                           0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000 );

mat4 g_matReflect2 = mat4( 0.7935718298, -0.0946179554, 0.6010749936, 0.0000000000,
                          -0.0946179554, 0.9566311240, 0.2755074203, 0.0000000000,
                           0.6010749936, 0.2755074203, -0.7502027750, 0.0000000000,
                           0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000 );

mat4 g_matReflect3 = mat4(-0.7127467394, -0.5999681950, 0.3633601665, 0.0000000000,
                          -0.5999681950, 0.7898335457, 0.1272835881, 0.0000000000,
                           0.3633601665, 0.1272835881, 0.9229129553, 0.0000000000,
                           0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000 );

vec4 g_planeReflect1 = vec4( 0.0077987094, 0.6606628895, 0.7506421208, -0.0000000000 );

vec4 g_planeReflect2 = vec4( 0.3212694824, 0.1472563744, -0.9354685545, -0.0000000000 );

vec4 g_planeReflect3 = vec4( -0.9254043102, -0.3241653740, 0.1963250339, -0.0000000000 );

/////////////////////////////////////////////////////////////////////////////////////////

vec3 HSVtoRGB(float h, float s, float v) 
{
  return((clamp(abs(fract(h +vec3(0.,2./3.,1./3.))*2.-1.)*3.-1.,0.,1.)-1.)*s+1.)*v;
}

mat3 rot3xy( vec2 angle )
{
  vec2 c = cos( angle );
  vec2 s = sin( angle );
  return mat3( c.y,       -s.y,        0.0,
               s.y * c.x,  c.y * c.x, -s.x,
               s.y * s.x,  c.y * s.x,  c.x );
}

vec4 DE1( in vec4 v )
{
  float fR = dot( v.xyz, v.xyz );
  vec4 q;
  int k = 0;
  vec3 vO = vec3( 0.0, 0.0, 0.0 );

  for ( int i = 0; i < 32; i++ )
  {
    q = v*g_matIterator1;
    v.xyz = q.xyz;

    if ( dot( v, g_planeReflect1 ) < 0.0 )
    {
      q = v*g_matReflect1;
      v.xyz = q.xyz;
      vO.x += 1.0;
    }

    if ( dot( v, g_planeReflect2 ) < 0.0 )
    {
      q = v*g_matReflect2;
      v.xyz = q.xyz;
      vO.y += 1.0;
    }

    if ( dot( v, g_planeReflect3 ) < 0.0 )
    {
      q = v*g_matReflect3;
      v.xyz = q.xyz;
      vO.z += 1.0;
    }

    q = v*g_matIterator2;
    v.xyz = q.xyz;

    v.xyz = v.xyz*g_fScale;
    fR = dot( v.xyz, v.xyz );
    k = i;
  }
  return vec4( vO, ( sqrt( fR ) - 2.0 ) * pow( g_fScale, -float(k+1) ) );
}

//------------------------------------------------------

float time = 0.0;  
float fL = 1.0;

//------------------------------------------------------
vec4 renderRay (in vec3 rayOrig, in vec3 rayDir)
{
  rayDir = normalize( rayDir );

  const float fRadius = 2.0;
  float b = dot( rayDir, rayOrig ) * 2.0;
  float c = dot( rayOrig, rayOrig ) - fRadius*fRadius;
  float ac4 = 4.0 * c;
  float b2 = b*b;

  vec4 color = vec4(0,0,0,1);
  color.rgb = -rayDir*0.2+0.8;
  color.rgb = pow( color.rgb, vec3( 0.9, 0.8, 0.5 ) );
  color.rgb *= 1.0-fL;
  if ( b2 - ac4 <= 0.0 )  return color;

  float root = sqrt( b2-ac4 );
  float at1 = max(0.0, (( -b - root ) / 2.0));
  float at2 = ( -b + root ) / 2.0;

  float t = at1;
  vec4 v = vec4( rayOrig + rayDir * t, 1.0 );
  vec4 vDE = vec4( 0.0, 0.0, 0.0, 0.0 );
  float fEpsilon = 0.0;

  float fEpsilonHelper = 1.0 / iResolution.x;
    
  float count = 0.0;
  for ( int k = 0; k < 100; k++ )
  {
    vDE = DE1( v );
    t += vDE.w;
    v.xyz = rayOrig + rayDir * t;

    fEpsilon = fEpsilonHelper * t;
		
    if ( vDE.a < fEpsilon ) 
    {
        count = float(k);
        break;
    }
    if ( t > at2 )     return color;
  }
    
  // colorizing by distance of fractal
  color.rgb = HSVtoRGB(count/25., 1.0-count/50., 0.8);
    
  vec4 vOffset = vec4( fEpsilon*1.8, 0.0, 0.0, 0.0 );
  vec4 vNormal = vec4(0.0);
  vNormal.x = DE1( v + vOffset.xyzw ).w - DE1( v - vOffset.xyzw ).w;
  vNormal.y = DE1( v + vOffset.yxzw ).w - DE1( v - vOffset.yxzw ).w;
  vNormal.z = DE1( v + vOffset.zyxw ).w - DE1( v - vOffset.zyxw ).w;
  vNormal.xyz = normalize( vNormal.xyz );

  vec4 vReflect = vec4(0.7);
  vReflect.xyz = reflect( rayDir, vNormal.xyz );

  vec2 vOccRefl = vec2( 0.0, 0.4 );
  
  float fMul = 2.0;
  float fMulMul = pow( 2.0, 9.0/10.0 ) * pow( fEpsilon, 1.0/10.0 ) * 0.5;
  float fW = 0.0;
  for ( int k = 0; k < 8; k++ )
  {
    vOccRefl.x += DE1( v + vNormal * fMul ).w / fMul;
    vOccRefl.y += DE1( v + vReflect * fMul ).w / fMul;
    fMul *= fMulMul;
  }
  vOccRefl /= 6.0;
  
  color.rgb *= vec3( vOccRefl.x * vOccRefl.y );
  color.rgb *= (vNormal.xyz*0.5+0.5)*(1.0-vOccRefl.x) +vec3(1.5)* vOccRefl.y;
  color.rgb = pow( color.rgb, vec3( 0.4, 0.5, 0.6 ) );
  color.rgb *= 1.0-fL;
  return vec4(color.rgb, 1.0);
}

//------------------------------------------------------
void mainVR (out vec4 fragColor, in vec2 fragCoord
            ,in vec3 fragRayOri, in vec3 fragRayDir)
{
  vec2 uv = (fragCoord - iResolution.xy*0.5) / iResolution.x;
  fL = length( uv );
    
  float tr = (.5 + .5 * sin(iTime)) * .2;
  g_fScale -= (tr*2.3 - 2.0) / 18.0;    // fractal scaling
    
  fragColor = renderRay (fragRayOri, fragRayDir);
}

//------------------------------------------------------
vec4 render(in vec2 pos)
{
  time = iTime * 0.1;  
  vec2 mouse = iMouse.xy / iResolution.xy;
  vec3 rayOrig = vec3( -3.0 - sin( time ), 0.0, 0.0 );
  vec2 uv = (pos - iResolution.xy*0.5) / iResolution.x;
  fL = length( uv );
  uv /= fL;
  uv *= 1.0-pow( 1.0-fL, 0.7 );
  vec3 rayDir = vec3(0.45+mouse.y, uv );

  mat3 rot = rot3xy( vec2( 0.0, time + mouse.x * 4.0) );
  rayDir  = rot * rayDir;
  rayOrig = rot * rayOrig;
    
  return renderRay (rayOrig, rayDir);
}

//------------------------------------------------------

#define AAX 1   // supersampling level. Make higher for more quality.
#define AAY 1   

float AA = float(AAX * AAY);

//------------------------------------------------------
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  float tr = (.5 + .5 * sin(iTime)) * .2;
  g_fScale -= (tr*2.3 - 2.0) / 18.0;    // fractal scaling

  if (AAX>1 || AAY>1)
  {
    vec4 col = vec4(0,0,0,1);
    for (int xp = 0; xp < AAX; xp++)
    for (int yp = 0; yp < AAY; yp++)
    {
      vec2 pos = fragCoord + vec2(xp,yp) / vec2(AAX,AAY);
      col += render (pos);
    }
    fragColor.rgb = col.rgb / AA;
  }
  else fragColor = render (fragCoord);
}
