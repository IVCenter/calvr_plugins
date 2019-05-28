using namespace glm;
inline vec3 getPlaneNormal(vec3 a, vec3 b, vec3 c){
    return normalize(cross(b-a, c-a));
}
inline bool PointInsideBoundingBox(vec3 p, vec3 aabb_min, vec3 aabb_max){
    if(p.x < aabb_min.x || p.x > aabb_max.x
       ||p.y < aabb_min.y || p.y > aabb_max.y
       ||p.z < aabb_min.z || p.z > aabb_max.z) return false;
    return true;
}

//ray intersect a plane
inline bool getRayIntersectionPlanePoint(vec3 ray_start, vec3 ray_dir, vec3 plane_point, vec3 plane_norm, vec3& out_point){
    float ray_plane_angle  = dot(ray_dir, plane_norm);
    if(ray_plane_angle == .0f)
        return false;
    //AX + BY + CZ + D = 0
    //D = -(PLANE_POINT * PLANE_NORM)
    //N = RAY_DIR * PLANE_NORM
    out_point =  ray_start - (( dot(plane_norm, ray_start) - dot(plane_point, plane_norm)) / ray_plane_angle) * ray_dir;
    return true;
}

inline bool getNearestRayAABBIntersection(vec3 ray_start, vec3 ray_dir, vec3 aabbMin, vec3 aabbMax, vec3& nearest_p){
    vec3 pop = aabbMin, interset_p;
    bool is_intersect = false;
    int best_pid;
    float c_dist, best_dist = FLT_MAX;
    const vec3 face_norm[6] = {vec3(1,0,0), vec3(0,0,1),vec3(0,1,0),
                               vec3(-1,0,0), vec3(0,0,-1), vec3(0,-1,0)};
    for(int i=0; i<6; i++) {
        if (i == 3) pop = aabbMax;
        if (getRayIntersectionPlanePoint(ray_start, ray_dir, pop, face_norm[i], interset_p)
            && PointInsideBoundingBox(interset_p, aabbMin, aabbMax)){
            c_dist = glm::distance(ray_start, interset_p);
            if (c_dist < best_dist) {
                c_dist = best_dist;
                best_pid = i;
                nearest_p = interset_p;
                is_intersect = true;
            }
        }
    }
    return is_intersect;
}

/************************************
 * getIntersectionPolygon: get all the points that form the polygon of plane-aabb intersection.
 * points starting from minimal x then y then z value
 * @param p point on plane, p_normal:surface normal of plane
 * @param aabb_min
 * @param aabb_max
 * @param polygon
 */
void getIntersectionPolygon(vec3 p, vec3 p_norm, vec3 aabb_min, vec3 aabb_max, std::vector<Polygon>& polygon, PolygonMap& face_map){
    vec3 dir; // direction to test
    vec3 p_rp;//current answer
    int pid = 0;
    // Test edges along X axis, pointing right.
    dir = vec3(aabb_max.x - aabb_min.x, .0f, .0f);
    if(getRayIntersectionPlanePoint(aabb_min, dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[BACK].push_back(pid);
        face_map[BOTTOM].push_back(pid);
        pid++;
    }

    if(getRayIntersectionPlanePoint(vec3(aabb_min.x, aabb_max.y, aabb_min.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[BACK].push_back(pid);
        face_map[UP].push_back(pid);
        pid++;
    }
    if(getRayIntersectionPlanePoint(vec3(aabb_min.x, aabb_min.y, aabb_max.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[FRONT].push_back(pid);
        face_map[BOTTOM].push_back(pid);
        pid++;
    }

    if(getRayIntersectionPlanePoint(vec3(aabb_min.x, aabb_max.y, aabb_max.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[FRONT].push_back(pid);
        face_map[UP].push_back(pid);
        pid++;
    }

    // Test edges along Y axis, pointing up.
    dir = vec3(0.f, aabb_max.y - aabb_min.y, 0.f);
    if(getRayIntersectionPlanePoint(vec3(aabb_min.x, aabb_min.y, aabb_min.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[BACK].push_back(pid);
        face_map[LEFT].push_back(pid);
        pid++;
    }
    if(getRayIntersectionPlanePoint(vec3(aabb_max.x, aabb_min.y, aabb_min.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[BACK].push_back(pid);
        face_map[RIGHT].push_back(pid);
        pid++;
    }
    if(getRayIntersectionPlanePoint(vec3(aabb_min.x, aabb_min.y, aabb_max.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[FRONT].push_back(pid);
        face_map[LEFT].push_back(pid);
        pid++;
    }
    if(getRayIntersectionPlanePoint(vec3(aabb_max.x, aabb_min.y, aabb_max.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[FRONT].push_back(pid);
        face_map[RIGHT].push_back(pid);
        pid++;
    }

    // Test edges along Z axis, pointing forward.
    dir = vec3(0.f, 0.f, aabb_max.z - aabb_min.z);
    if(getRayIntersectionPlanePoint(vec3(aabb_min.x, aabb_min.y, aabb_min.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[LEFT].push_back(pid);
        face_map[BOTTOM].push_back(pid);
        pid++;
    }
    if(getRayIntersectionPlanePoint(vec3(aabb_max.x, aabb_min.y, aabb_min.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[BOTTOM].push_back(pid);
        face_map[RIGHT].push_back(pid);
        pid++;
    }
    if(getRayIntersectionPlanePoint(vec3(aabb_min.x, aabb_max.y, aabb_min.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[UP].push_back(pid);
        face_map[LEFT].push_back(pid);
        pid++;
    }
    if(getRayIntersectionPlanePoint(vec3(aabb_max.x, aabb_max.y, aabb_min.z), dir, p, p_norm, p_rp)
       && PointInsideBoundingBox(p_rp, aabb_min, aabb_max)){
        polygon.push_back(std::make_pair(p_rp, pid));
        face_map[UP].push_back(pid);
        face_map[RIGHT].push_back(pid);
        pid++;
    }

    //sort the plane
    if(polygon.empty()) return;

    //left-bottom corner
    vec3 origin = polygon[0].first;//findMinVec3(polygon);//polygon[0];

    std::sort(polygon.begin(), polygon.end(), [&](const Polygon& p1, const Polygon& p2)->bool{
        vec3 cross_v = glm::cross(p1.first - origin, p2.first - origin);
        return dot(cross_v, p_norm)<0;
    });
}