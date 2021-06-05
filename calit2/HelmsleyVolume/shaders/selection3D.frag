#version 460

out vec4 FragColor;

in vs_out {
    vec3 localPos;
} i;

uniform vec3 Start;
uniform vec3 End;

void main() 
{
        vec3 border = vec3(0,0,0);
        vec3 position = i.localPos;
        vec4 color = vec4(0);
        //position += .5;
        position += .5;

        //position.z+= .5;
        //if(position.z > .40 && position.x > .4)
        if(position.z > .95 && position.x > .95)
            color = vec4(border, 1.0);
        else if(position.z > .95 && position.x < .05)
            color = vec4(border, 1.0);
        else if(position.z > .95 && position.y > .95)
            color = vec4(border, 1.0);
        else if(position.z > .95 && position.y < .05)
            color = vec4(border, 1.0);

        else if(position.x > .95 && position.y > .95)
            color = vec4(border, 1.0);
        else if(position.x > .95 && position.y < .05)
            color = vec4(border, 1.0);

        else if(position.x > .95 && position.z > .95)
            color = vec4(border, 1.0);
        else if(position.x > .95 && position.z < .05)
            color = vec4(border, 1.0);

        else if(position.y > .95 && position.x > .95)
            color = vec4(border, 1.0);
        else if(position.y > .95 && position.x < .05)
            color = vec4(border, 1.0);

        else if(position.y > .95 && position.z > .95)
            color = vec4(border, 1.0);
        else if(position.y > .95 && position.z < .05)
            color = vec4(border, 1.0);

        else if(position.y < .05 && position.z < .05)
            color = vec4(border, 1.0);
        else if(position.y < .05 && position.x < .05)
            color = vec4(border, 1.0);
        else if(position.z < .05 && position.x < .05)
            color = vec4(border, 1.0);
        else
            color = vec4(1,0,0, 0.51);

        if(position.x < .85 && position.x > .15 && position.y < .85 && position.y > .15)
            discard;
    
        FragColor = color;
}