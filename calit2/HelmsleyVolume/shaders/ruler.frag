#version 460

out vec4 FragColor;

in vs_out {
    vec3 worldPosition;
} i;

uniform vec3 Start;
uniform vec3 End;

void main() {
    vec3 vector = End - Start;
    vec3 posvector = i.worldPosition - Start;
    vec3 direction = normalize(vector);
    //float length = length(vector);

    vec3 projected = dot(posvector, direction) * direction;
    vec3 nonprojected = posvector - projected;
    float projlength = length(projected);
    float nonlength = length(nonprojected);

    vec4 marks = vec4(0,0,0,0.9);

    if(nonlength > 3 && mod(projlength + 0.8, 100) < 1.6)
    {
        FragColor = marks;
    }
    else if(nonlength > 5 && mod(projlength + 0.6, 50) < 1.2)
    {
        FragColor = marks;
    }
    else if(nonlength > 7 && mod(projlength + 0.4, 10) < 0.8)
    {
        FragColor = marks;
    }
    //else if(nonlength > 6 && mod(projlength + 0.25, 1) < 0.5)
    //{
    //    FragColor = vec4(0,0,0,1);
    //}
    else
    {
        FragColor = vec4(1,1,1,0.5);
    }
}