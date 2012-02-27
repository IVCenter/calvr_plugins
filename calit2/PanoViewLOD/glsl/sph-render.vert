
uniform int level;

uniform vec3 pos_a;
uniform vec3 pos_b;
uniform vec3 pos_c;
uniform vec3 pos_d;

uniform vec2 tex_a[8];
uniform vec2 tex_d[8];

//------------------------------------------------------------------------------

vec3 slerp(vec3 a, vec3 b, float k)
{
    float l = 1.0 - k;
    float O = acos(dot(a, b));
    
    return a * sin(l * O) / sin(O)
         + b * sin(k * O) / sin(O);
}

vec3 bislerp(vec3 a, vec3 b, vec3 c, vec3 d, vec2 k)
{
    vec3 t = slerp(a, b, k.x);
    vec3 u = slerp(c, d, k.x);

    return slerp(t, u, k.y);
}

//------------------------------------------------------------------------------

void main()
{
    vec2 t =  tex_a[level]
           + (tex_d[level] - tex_a[level]) * gl_Vertex.xy;

    vec3 v = bislerp(pos_a, pos_b, pos_c, pos_d, t);

    gl_TexCoord[0].xy = t;

    gl_Position = gl_ModelViewProjectionMatrix * vec4(v, 1.0);
}
