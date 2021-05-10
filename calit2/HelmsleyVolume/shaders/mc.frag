#version 460

out vec4 FragColor;

in vs_out {
	vec2 uv;
	vec3 vertPos;
	vec3 norm;
	vec3 fragPos;
} i;

uniform vec3 lightPos;

vec3 CalcPointLight(vec3 lightColor, vec3 lightPos);
vec3 ColonTexture();
float hash1(uint n);

void main() {
    float numLights = 1;
	vec3 lightColor = vec3(1.0);
	vec3 lightPosMult[] = {
        vec3(10.0, 5.0, 0.0),
        vec3(0.0, -5.0, 10.0),
        vec3(-10.0, 5.0, -10.0)
    };

    vec3 result = vec3(0.0);
    for(int i = 0; i < numLights; ++i) {
        //result += CalcPointLight(lightColor, lightPos[i]);
        result += CalcPointLight(lightColor, lightPos);
    }

    // attenuation
    //float dist = length(lightPos - i.fragPos);
    //float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    

    
        
    FragColor = vec4(result, 1.0);
}

vec3 CalcPointLight(vec3 lightColor, vec3 lightPos) {
   //MVP
	vec3 viewPos = vec3(0.0);

	//Material
	vec3 matAmbient = vec3(0.25, 0.15, 0.15);
	vec3 matDiffuse = vec3(1.0, 0.42, 0.42);
	vec3 matSpecular = vec3(0.75, 0.4, 0.4);
    float shininess = 30.0;

	// ambient
    vec3 ambient = lightColor * matAmbient;
  	
    // diffuse 
    vec3 norm = normalize(i.norm);
    vec3 lightDir = normalize(lightPos - i.fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    //vec3 diffuse = lightColor * (diff * matDiffuse);
    vec3 diffuse = ColonTexture();
    
    // specular
    vec3 viewDir = normalize(viewPos - i.fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = lightColor * (spec * matSpecular);  

    // texture (noise)
    vec3 tex = ColonTexture();
    return normalize(ambient + diffuse + specular);// + tex);
}

vec3 ColonTexture() {
    //vec3 result = vec3( hash1(uint(i.vertPos.x + i.vertPos.z)));
	vec3 ratio = vec3(1, 0.42, 0.42);

    float rand = hash1(uint(i.vertPos.x * i.vertPos.y * i.vertPos.z));
    //float scale = ((((rand + 1) / 2) * 105) + 150);
    float scale = ((rand + 1) / 2);

    vec3 result = ratio * scale;

    return result;
}

float hash1( uint n ) {
	n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float( n & uvec3(0x7fffffffU))/float(0x7fffffff);
}


