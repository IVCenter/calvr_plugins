#extension GL_EXT_gpu_shader4 : enable

void main()
{
    gl_FragData[0] = gl_Color;
    float depth = 1.0 - gl_FragCoord.z;
    unsigned int depthi = uint(float(16777215) * depth);
    unsigned int depthsmall = depthi & 0xFF;
    gl_FragData[2].r = float(depthsmall) / 255.0;
    depthi = depthi - depthsmall;
    gl_FragData[1].r = float(depthi) / 16776960.0;
    //gl_FragData[1].r = 1.0 - gl_FragCoord.z;
    //gl_FragData[1].r = 1.0;
}
