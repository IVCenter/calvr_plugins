// simple fragment shader

varying float LightIntensity;

void
main()
{
    gl_FragColor = vec4( LightIntensity*gl_Color.rgb, 1. );
}
