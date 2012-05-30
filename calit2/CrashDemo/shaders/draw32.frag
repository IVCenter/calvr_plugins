void main()
{
    gl_FragData[0] = gl_Color;
    gl_FragData[1].r = gl_FragCoord.z;
    //gl_FragData[1].r = 1.0;
}
