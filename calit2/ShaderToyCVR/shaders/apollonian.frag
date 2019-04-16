// Created by inigo quilez - iq/2016
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.


// Antialiasing level. Make it 1 if you have a slow machine
#define AA 1

vec3 hash3( vec2 p )
{
    vec3 q = vec3( dot(p,vec2(127.1,311.7)), 
				   dot(p,vec2(269.5,183.3)), 
				   dot(p,vec2(419.2,371.9)) );
	return fract(sin(q)*43758.5453);
}

float iqnoise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
		
	float k = 1.0+63.0*pow(1.0-v,4.0);
	
	float va = 0.0;
	float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = vec2( float(i),float(j) );
		vec3 o = hash3( p + g )*vec3(u,u,1.0);
		vec2 r = g - f + o.xy;
		float d = dot(r,r);
		float ww = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
		va += o.z*ww;
		wt += ww;
    }
	
    return va/wt;
}

vec4 ptexture(vec2 uv) {
	return vec4(iqnoise(uv, 0.5, .5));
}

vec3 map( vec3 p )
{
	float scale = 1.0;
    
    float orb = 10000.0;

    for( int i=0; i<6; i++ )
	{
		p = -1.0 + 2.0*fract(0.5*p+0.5);

        p -= sign(p)*0.04; // trick
        
        float r2 = dot(p,p);
		float k = 0.95/r2;
		p     *= k;
		scale *= k;

        orb = min( orb, r2);
	}

    float d1 = sqrt( min( min( dot(p.xy,p.xy), dot(p.yz,p.yz) ), dot(p.zx,p.zx) ) ) - 0.02;
    float d2 = abs(p.y);
    float dmi = d2;
    float adr = 0.7*floor((0.5*p.y+0.5)*8.0);
    if( d1<d2 )
    {
        dmi = d1;
        adr = 0.0;
    }
    return vec3( 0.5*dmi/scale, adr, orb );
}

vec3 trace( in vec3 ro, in vec3 rd )
{
	float maxd = 20.0;
    float t = 0.01;
    vec2  info = vec2(0.0);
    for( int i=0; i<256; i++ )
    {
	    float precis = 0.001*t;
        
        vec3  r = map( ro+rd*t );
	    float h = r.x;
        info = r.yz;
        if( h<precis||t>maxd ) break;
        t += h;
    }

    if( t>maxd ) t=-1.0;
    return vec3( t, info );
}

vec3 calcNormal( in vec3 pos, in float t )
{
    float precis = 0.0001 * t * 0.57;

    vec2 e = vec2(1.0,-1.0)*precis;
    return normalize( e.xyy*map( pos + e.xyy ).x + 
					  e.yyx*map( pos + e.yyx ).x + 
					  e.yxy*map( pos + e.yxy ).x + 
                      e.xxx*map( pos + e.xxx ).x );
}

vec3 forwardSF( float i, float n) 
{
    const float PI  = 3.141592653589793238;
    const float PHI = 1.618033988749894848;
    float phi = 2.0*PI*fract(i/PHI);
    float zi = 1.0 - (2.0*i+1.0)/n;
    float sinTheta = sqrt( 1.0 - zi*zi);
    return vec3( cos(phi)*sinTheta, sin(phi)*sinTheta, zi);
}

float calcAO( in vec3 pos, in vec3 nor )
{
	float ao = 0.0;
    for( int i=0; i<16; i++ )
    {
        vec3 w = forwardSF( float(i), 16.0 );
		w *= sign( dot(w,nor) );
        float h = float(i)/15.0;
        ao += clamp( map( pos + nor*0.01 + w*h*0.15 ).x*2.0, 0.0, 1.0 );
    }
	ao /= 16.0;
	
    return clamp( ao*16.0, 0.0, 1.0 );
}


vec3 textureBox( in vec3 pos, in vec3 nor )
{
    vec3 w = nor*nor;
    return (w.x*ptexture( pos.yz ).xyz + 
            w.y*ptexture( pos.zx ).xyz + 
            w.z*ptexture( pos.xy ).xyz ) / (w.x+w.y+w.z);
}

vec3 render( in vec3 ro, in vec3 rd )
{
    vec3 col = vec3(0.0);
    vec3 res = trace( ro, rd );;
    float t = res.x;
    if( t>0.0 )
    {
        vec3  pos = ro + t*rd;
        vec3  nor = calcNormal( pos, t );
        float fre = clamp(1.0+dot(rd,nor),0.0,1.0);
        float occ = pow( clamp(res.z*2.0,0.0,1.0), 1.2 );
              occ = 1.5*(0.1+0.9*occ)*calcAO(pos,nor);        
        vec3  lin = vec3(1.0,1.0,1.5)*(2.0+fre*fre*vec3(1.8,1.0,1.0))*occ*(1.0-0.5*abs(nor.y));
        
      	col = 0.5 + 0.5*cos( 6.2831*res.y + vec3(0.0,1.0,2.0) );  
        col *= textureBox( pos, nor ).xyz;
        col = col*lin;
        col += 0.6*pow(1.0-fre,32.0)*occ*vec3(0.5,1.0,1.5);        
        col *= exp(-0.3*t);
    }
    col.z += 0.01;

    return sqrt(col);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float time = iTime*0.15 + 0.005*iMouse.x;
    
    vec3 tot = vec3(0.0);
    #if AA>1
    for( int jj=0; jj<AA; jj++ )
    for( int ii=0; ii<AA; ii++ )
    #else
    int ii = 0, jj = 0;
    #endif
    {
        vec2 q = fragCoord+vec2(float(ii),float(jj))/float(AA);

        // camera
        vec3 ro = vec3( 2.8*cos(0.1+.33*time), 0.5 + 0.20*cos(0.37*time), 2.8*cos(0.5+0.35*time) );
        vec3 ta = vec3( 1.9*cos(1.2+.41*time), 0.5 + 0.10*cos(0.27*time), 1.9*cos(2.0+0.38*time) );
        float roll = 0.2*cos(0.1*time);
        vec3 cw = normalize(ta-ro);
        vec3 cp = vec3(sin(roll), cos(roll),0.0);
        vec3 cu = normalize(cross(cw,cp));
        vec3 cv = normalize(cross(cu,cw));

        #if 1
        vec2 p = (2.0*q-iResolution.xy)/iResolution.y;
        vec3 rd = normalize( p.x*cu + p.y*cv + 2.0*cw );
        #else
        vec2 p = q/iResolution.xy;
        vec2 an = 3.1415926535898 * (p*vec2(2.0, 1.0) - vec2(0.0,0.5));
        vec3 rd = vec3(cos(an.y) * sin(an.x), sin(an.y), cos(an.y) * cos(an.x));
		#endif

        tot += render( ro, rd );
    }
    
    tot = tot/float(AA*AA);
    
	fragColor = vec4( tot, 1.0 );	

}

void mainVR( out vec4 fragColor, in vec2 fragCoord, in vec3 fragRayOri, in vec3 fragRayDir )
{
    vec3 col = render( fragRayOri + vec3(0.82,1.3,-0.3), fragRayDir );
    fragColor = vec4( col, 1.0 );
}