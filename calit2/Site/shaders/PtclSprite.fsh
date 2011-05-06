uniform sampler2D baseTexture;

void main (void)
{
    // Note: That's defined in the PointSprite wrapper
    // (GLSL 120 will support gl_PointCoord?)
    vec2 tex_coord = gl_TexCoord[0].xy;
    tex_coord.y = 1.0-tex_coord.y;
    
    // use color alpha to look up corresponding color
    vec4 result = texture2D(baseTexture, vec2(gl_Color.a));

    /////////////////////////
    // compute alpha value //
    /////////////////////////
    float d = 2.0*distance(tex_coord.xy, vec2(0.5, 0.5));
    result.a = step(d, 1.0);

    // Phong
    // -----
    vec3 eye_vector = normalize(vec3(0.0, 0.0, 1.0));
    vec3 light_vector = normalize(vec3(2.0, 2.0, 1.0));
    vec3 surface_normal = normalize(vec3(2.0*
        (tex_coord.xy-vec2(0.5, 0.5)), sqrt(1.0-d)));
    vec3 half_vector = normalize(eye_vector+light_vector);

    // Calculate specular component
    float specular = dot(surface_normal, half_vector);
    float diffuse  = dot(surface_normal, light_vector);

    // Use the lit function to compute lighting vector from diffuse and
    // specular values (does not work with GLSL at the moment)
    // vec4 lighting = lit(diffuse, specular, 8.0);
    vec4 lighting = vec4(0.75, max(diffuse, 0.0), pow(max(specular, 0.0), 40.0), 0.0);

    //result.rgb = lighting.x*vec3(0.2, 0.8, 0.2)+lighting.y*vec3(0.6, 0.6, 0.6)+
    //    lighting.z*vec3(0.25, 0.25, 0.25);

    // TODO need to adjust this
    result.rgb = lighting.x*vec3(0.2, 0.2, 0.2) + lighting.y * result.xyz +
        lighting.z * vec3(0.25, 0.25, 0.25);


    gl_FragColor = result;
}
