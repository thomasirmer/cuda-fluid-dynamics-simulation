#include <math.h>
#include <float.h>

// ****************************************************************************
// DEVICE FUNCTIONS
// ****************************************************************************

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

	// normalize vectors
	float lenght1 = sqrt(x1 * x1 + y1 * y1) + FLT_EPSILON;
	float normX1 = x1 / lenght1;
	float normY1 = y1 / lenght1;

	float lenght2 = sqrt(x2 * x2 + y2 * y2) + FLT_EPSILON;
	float normX2 = x2 / lenght2;
	float normY2 = y2 / lenght2;

	// calculate angle
	float angle = (atan2(normY2, normX2) - atan2(normY1, normX1)) / M_PI * 180.0f;

	// correct angle at 180°-overflow
	if (angle < -180.0f)
		angle += 360.0f;
	if (angle > 180.0f)
		angle -= 360.0f;

	return angle;
}

// ****************************************************************************
// HOST FUNCTIONS
// ****************************************************************************

// vector calculations
float getVectorLengthSerial(float xCoord, float yCoord) {
	return (float) sqrt(xCoord * xCoord + yCoord * yCoord);
}

// return the angle of the vector
float getVectorAngleSerial(float xCoord, float yCoord) {
	float angleRad = atan2(yCoord, xCoord);
	float angleDeg = (angleRad / M_PI) * 180.0f;
	return angleDeg;
}

float getAngleBetweenSerial(float x1, float y1, float x2, float y2) {

	// normalize vectors
	float lenght1 = sqrt(x1 * x1 + y1 * y1) + FLT_EPSILON;
	float normX1 = x1 / lenght1;
	float normY1 = y1 / lenght1;

	float lenght2 = sqrt(x2 * x2 + y2 * y2) + FLT_EPSILON;
	float normX2 = x2 / lenght2;
	float normY2 = y2 / lenght2;

	// calculate angle
	float angle = (atan2(normY2, normX2) - atan2(normY1, normX1)) / M_PI * 180.0f;

	// correct angle at 180°-overflow
	if (angle < -180.0f)
		angle += 360.0f;
	if (angle > 180.0f)
		angle -= 360.0f;

	return angle;
}
