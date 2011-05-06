varying float light;

void main()
{
    vec4 color = gl_Color * light;
    gl_FragData[0] = color;
    gl_FragData[1].r = 1.0 - gl_FragCoord.z;
    //gl_FragData[1].r = 1.0;
}
