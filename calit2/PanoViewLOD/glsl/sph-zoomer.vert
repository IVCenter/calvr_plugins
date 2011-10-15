
uniform int level;

uniform vec3 pos_a;
uniform vec3 pos_b;
uniform vec3 pos_c;
uniform vec3 pos_d;

uniform vec2 tex_a[8];
uniform vec2 tex_d[8];

uniform vec3  zoomv;
uniform float zoomk;

//------------------------------------------------------------------------------

vec3 slerp(vec3 a, vec3 b, float k)
{
    float l = 1.0 - k;
    float O = acos(dot(a, b));
    
    if (O > 0.0)
        return a * sin(l * O) / sin(O)
             + b * sin(k * O) / sin(O);
    else
        return a;
}

vec3 bislerp(vec3 a, vec3 b, vec3 c, vec3 d, vec2 k)
{
    vec3 t = slerp(a, b, k.x);
    vec3 u = slerp(c, d, k.x);

    return slerp(t, u, k.y);
}

float scale(float k, float t)
{
    if (k < 1.0)
        return min(t / k, 1.0 - (1.0 - t) * k);
    else
        return max(t / k, 1.0 - (1.0 - t) * k);
}

vec3 zoom(vec3 v)
{
    const float pi = 3.1415927;
    
    float a = acos(dot(v, zoomv));
    
    if (a > 0.0)
    {
        float b = scale(zoomk, a / pi) * pi;

        vec3 y = normalize(cross(v, zoomv));
        vec3 x = normalize(cross(zoomv, y));
        
        return zoomv * cos(b) + x * sin(b);
    }
    else return v;
}

//vec3 zoom(vec3 v)
//{
//    const float pi = 3.1415927;
//    
//    float a = acos(dot(v, zoomv));
//    
//    if (a > 0.0)
//    {
//        float b = clamp(a * zoomk, 0.0, pi);
//
//        vec3 y = normalize(cross(v, zoomv));
//        vec3 x = normalize(cross(zoomv, y));
//        
//        return zoomv * cos(b) + x * sin(b);
//    }
//    else return v;
//}

//------------------------------------------------------------------------------

void main()
{
    vec2 t =  tex_a[level]
           + (tex_d[level] - tex_a[level]) * gl_Vertex.xy;

    vec3 v = normalize(bislerp(pos_a, pos_b, pos_c, pos_d, t));

    v = zoom(v);

    gl_TexCoord[0].xy = t;

    gl_Position = gl_ModelViewProjectionMatrix * vec4(v, 1.0);
}
