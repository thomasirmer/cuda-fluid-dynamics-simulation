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

__device__ float getAngleBetween(float x1, float y1, float x2, float y2) {
	const float EPS = 1.0e-3;

	float angle1 = (atan2(y1, x1) / M_PI) * 180.0f;
	float angle2 = (atan2(y2, x2) / M_PI) * 180.0f;
	float angle = angle2 - angle1;

	if ((abs(angle) < 270.0f + EPS && abs(angle) > 270.0f - EPS) || (abs(angle) < 90.0f + EPS && abs(angle) > 90.0f - EPS))
		angle = 90;
	else {
		int c = abs(angle) / 90.0f;
		angle = abs(angle) - c * 90.0f;
	}
	return angle;
}
