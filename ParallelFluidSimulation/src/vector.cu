// vector calculations
__device__ float getVectorLength(float xCoord, float yCoord) {
	return (float) sqrt(xCoord * xCoord + yCoord * yCoord);
}

// return the angle of the vector
__device__ float getVectorAngle(float xCoord, float yCoord) {
	float angleRad = atan2(yCoord, xCoord);
	float angleDeg = (angleRad / M_PI) * 180.0f;
	return angleDeg;
}
