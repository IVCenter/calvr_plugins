#version 460

out vec4 FragColor;

in vs_out {
	vec2 uv;
	vec3 vertPos;
	vec3 norm;
	vec3 fragPos;
} i;

void main() {
    //MVP
	vec3 viewPos = vec3(0.0);

    //Light
	vec3 lightColor = vec3(1.0);
    vec3 lightPos = vec3(0.0, 10.0, 0.0);

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
    vec3 diffuse = lightColor * (diff * matDiffuse);
    
    // specular
    vec3 viewDir = normalize(viewPos - i.fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = lightColor * (spec * matSpecular);  
    
        
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}


