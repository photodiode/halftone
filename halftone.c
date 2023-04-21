
#include <stdio.h>
#include <stdlib.h>

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include <math.h>
#include <time.h>


// gcc -std=c11 -O2 -lm halftone.c -o halftone && ./halftone


// math
#define PI 3.14159265358979323846

static inline float signf(float f) {
	return (f < 0.0f) ? -1.0f : 1.0f;
}

static inline float clampf(float x, float min, float max) {
	x = fmaxf(min, x);
	x = fminf(max, x);
	return x;
}

static inline float smoothstepf(float x, float edge0, float edge1) {
	x = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
	return x * x * (3.0f - 2.0f * x);
}

static inline float lerpf(float a, float b, float f) {
	return (a * (1.0f - f)) + (b * f);
}

float fractf(float x) {
	return x - floorf(x);
}
// ----


// vector
typedef union {
	struct { float x, y; };
	struct { float w, h; };
} vec;

#define VEC(X, Y) (vec){{X, Y}}

#define VEC_ZERO VEC(0, 0)
#define VEC_ONE  VEC(1, 1)


static inline float vec_dot(vec a, vec b) {
	return a.x * b.x + a.y * b.y;
}

static inline float vec_length(vec v) {
	return sqrtf(v.x * v.x + v.y * v.y);
}

static inline vec vec_add(vec a, vec b) {
	return VEC(
		a.x + b.x,
		a.y + b.y
	);
}

static inline vec vec_sub(vec a, vec b) {
	return VEC(
		a.x - b.x,
		a.y - b.y
	);
}

static inline vec vec_subs(vec a, float b) {
	return VEC(
		a.x - b,
		a.y - b
	);
}

static inline vec vec_mul(vec a, vec b) {
	return VEC(
		a.x * b.x,
		a.y * b.y
	);
}

static inline vec vec_scale(vec v, float s) {
	return VEC(
		v.x * s,
		v.y * s
	);
}

static inline vec vec_div(vec a, vec b) {
	return VEC(
		a.x / b.x,
		a.y / b.y
	);
}

static inline vec vec_divs(vec v, float s) {
	return VEC(
		v.x / s,
		v.y / s
	);
}

static inline vec vec_mod(vec a, float m) {
	return VEC(
		fmodf(a.x, m),
		fmodf(a.y, m)
	);
}

static inline vec vec_min(vec a, vec b) {
	return VEC(
		fminf(a.x, b.x),
		fminf(a.y, b.y)
	);
}

static inline vec vec_max(vec a, vec b) {
	return VEC(
		fmaxf(a.x, b.x),
		fmaxf(a.y, b.y)
	);
}

static inline vec vec_abs(vec v) {
	return VEC(
		fabsf(v.x),
		fabsf(v.y)
	);
}

static inline vec vec_floor(vec v) {
	return VEC(
		floorf(v.x),
		floorf(v.y)
	);
}

static inline vec vec_round(vec v) {
	return VEC(
		roundf(v.x),
		roundf(v.y)
	);
}

static inline float vec_hash(vec p) {
	p = VEC(
		50.0f * fractf(p.x * 0.3183099f),
		50.0f * fractf(p.y * 0.3183099f)
	);
	return fractf(p.x*p.y*(p.x+p.y));
}

static inline float vec_noise(vec x) {
	vec p = VEC(floorf(x.x), floorf(x.y));
	vec w = VEC(fractf(x.x), fractf(x.y));

	vec z = vec_scale(w, 2);
	vec u = vec_mul(vec_mul(w, w), VEC(3.0f - z.x, 3.0f - z.y));

	float a = vec_hash(vec_add(p, VEC(0, 0)));
	float b = vec_hash(vec_add(p, VEC(1, 0)));
	float c = vec_hash(vec_add(p, VEC(0, 1)));
	float d = vec_hash(vec_add(p, VEC(1, 1)));

	return -1.0+2.0*(a + (b-a)*u.x + (c-a)*u.y + (a - b - c + d)*u.x*u.y);
}

static inline float blerpf(float a, float b, float c, float d, vec v) {
	float i = lerpf(a, d, v.x);
	float k = lerpf(b, c, v.x);

	return lerpf(i, k, v.y);
}
// ----


// matrix
typedef union {
	vec   vec[2];
	float raw[2][2];
} mat;

#define MAT(A, B, C, D) (mat){VEC(A, B), VEC(C, D)}
#define MAT_ZERO (mat){VEC_ZERO, VEC_ZERO}
#define MAT_IDENTITY (mat){VEC(1.0f, 0.0f), VEC(0.0f, 1.0f)}


static inline vec mat_mulv(mat m, vec v) {
	return VEC(
		m.raw[0][0] * v.x + m.raw[1][0] * v.y,
		m.raw[0][1] * v.x + m.raw[1][1] * v.y
	);
}

static inline mat mat_inv(mat m) {
	return MAT(
		m.raw[0][0],
		-m.raw[0][1],
		-m.raw[1][0],
		m.raw[1][1]
	);
}
// ----


// sdf
typedef float sdf;


static inline vec sd_move(vec p, vec a) {
	return vec_sub(p, a);
}

static inline vec sd_repeat(vec p, float c) {
	return vec_subs(vec_mod(vec_abs(p), c), 0.5f*c);
}

static inline sdf sd_circle(vec p, float r) {
	return vec_length(p) - r;
}

static inline float sd_clamp(sdf d, float s) {
	//return 0.5f - clampf(d*s, -0.5, 0.5);
	return smoothstepf(d*s, 0.45, -0.35);
}

float sd_noise(vec x) {
	float n = vec_noise(x);
	float a = 0.5f * n;

	return a;
}

static inline float dot_fill(float i) {
	if (i < 0.0001) return -1;
	float h = i * 0.5f;
	return sqrtf(2.0f*h*h);
}

static inline sdf sd_halftone(vec p, mat m, float d) {

	vec q = mat_mulv(m, p);
	    q = sd_repeat(q, 1);

	sdf circle = sd_circle(q, dot_fill(d));

	return circle;
}
// ----


// image
typedef struct {
	uint8_t r, g, b;
} rgb;
#define RGB(R, G, B) (rgb){R, G, B}

typedef struct {
	float c, m, y, k;
} cmyk;
#define CMYK(C, M, Y, K) (cmyk){C, M, Y, K}

static inline cmyk rgb_to_cmyk(rgb input) {

	float r = (float)input.r / 255.0f;
	float g = (float)input.g / 255.0f;
	float b = (float)input.b / 255.0f;

	cmyk output;

	output.k = 1.0f - fmaxf(fmaxf(r, g), b);
	output.c = (1.0f - r - output.k) / (1.0f - output.k);
	output.m = (1.0f - g - output.k) / (1.0f - output.k);
	output.y = (1.0f - b - output.k) / (1.0f - output.k);

	return output;
}

static inline rgb cmyk_to_rgb(cmyk input) {
	return RGB(
		255.0f * (1.0f - input.c) * (1.0f - input.k),
		255.0f * (1.0f - input.m) * (1.0f - input.k),
		255.0f * (1.0f - input.y) * (1.0f - input.k)
	);
}

static inline cmyk cmyk_mix(cmyk a, cmyk b) {
	cmyk c = {
		fminf(1, a.c + b.c),
		fminf(1, a.m + b.m),
		fminf(1, a.y + b.y),
		fminf(1, a.k + b.k)
	};

	float g = fminf(fminf(c.c, c.m), c.y);
	c.c -= g;
	c.m -= g;
	c.y -= g;
	c.k = fminf(1, c.k + g);

	return c;
}


int ppm_read(const char* filename, uint32_t* w, uint32_t* h, void** data, size_t* size) {

	FILE* file = fopen(filename, "r");

	if(file == NULL) return -1;

	char line[16];

	fgets(line, 16, file); //magic number

	fgets(line, 16, file);
	*w = strtol(line, NULL, 10);
	char* next = strchr(line, ' ');
	*h = strtol(next, NULL, 10);

	fgets(line, 16, file);
	int max = strtol(line, NULL, 10);

	int bytes = (max < 256) ? 1 : 2;

	*size = *w * *h * 3 * bytes;

	*data = malloc(*size);

	fread(*data, *size, 1, file);

	fclose(file);

	return 0;
}


int ppm_write(char *filename, uint32_t w, uint32_t h, void *data, size_t size) {

	FILE *file = fopen(filename, "wb");
	if(file == NULL) return -1;

	fprintf(file, "P6\n%d %d\n%d\n", w, h, 255);

	fwrite(data, size, 1, file);
	fclose(file);

	return 0;
}
// ----


#define HT_MAT_C MAT(0.966, 0.259, -0.259, 0.966) // C  15
#define HT_MAT_M MAT(0.707, 0.707, -0.707, 0.707) // M  45
#define HT_MAT_Y MAT(1.000, 0.000, -0.000, 1.000) // Y   0
#define HT_MAT_K MAT(0.259, 0.966, -0.966, 0.259) // K  75


cmyk sample_pixel(cmyk* image, vec in, vec out, vec p, mat r, vec sf, float dot_size) {

	p = mat_mulv(r, p);

	vec s = vec_floor(vec_mul(sf, vec_div(p, out))); // sf * (p / out)

	mat ri = mat_inv(r);

	vec t = vec_mul(in, vec_div(s, sf)); // in * (s / sf)
	    t = mat_mulv(ri, t);

	vec k[5];
	k[0] = VEC( 0.5,  0.5);
	k[1] = VEC( 0.5, -0.5);
	k[2] = VEC(-0.5, -0.5);
	k[3] = VEC(-0.5,  0.5);
	k[4] = VEC( 0.0,  0.0);

	float in_dot_size = (dot_size / out.w) * in.w * 0.5;

	cmyk size = {0};

	for (int i = 0; i < 5; i++) {
		k[i] = vec_scale(k[i], in_dot_size);
		k[i] = vec_add(k[i], VEC(in_dot_size, in_dot_size));
		k[i] = mat_mulv(ri, k[i]);
		k[i] = vec_add(k[i], t);

		uint32_t ix = clampf(floorf(k[i].x), 0, in.w-1);
		uint32_t iy = clampf(floorf(k[i].y), 0, in.h-1);

		cmyk color = image[ix + iy * (uint32_t)in.w];

		size.c += color.c;
		size.m += color.m;
		size.y += color.y;
		size.k += color.k;
	}

	size.c /= 5;
	size.m /= 5;
	size.y /= 5;
	size.k /= 5;

	return size;
}


int main() {

	// file
	rgb*     file;
	size_t   file_size;
	uint32_t in_w, in_h;

	ppm_read("shop.ppm", &in_w, &in_h, (void**)&file, &file_size);
	// ----

	printf("%u x %u\n", in_w, in_h);

	// input
	cmyk*  img;
	size_t in_size = in_w * in_h * sizeof(*img);
	       img     = calloc(in_size, 1);

	for (uint32_t i = 0; i < in_w*in_h; i++) {
		img[i] = rgb_to_cmyk(file[i]);
	}

	free(file);
	// ----

	// output
	uint32_t width  = 800;
	uint32_t height = in_h * ((float)width / (float)in_w);

	rgb*   image;
	size_t image_size = width * height * sizeof(*image);
	       image      = calloc(image_size, 1);
	// ----

	float dot_size = 2.467; // size of dots
	float dot_size_inv = 1.0f / dot_size;

	vec sf = VEC(width * dot_size_inv, height * dot_size_inv);

	float smear = dot_size_inv;
	//float smear = 0.35f; // less is more blurry dots

	clock_t time_start = clock();
	clock_t time_end = 0;

	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {

			vec pc = VEC(x, y);
			vec pm = VEC(x, y);
			vec py = VEC(x, y);
			vec pk = VEC(x, y);

			cmyk wobble = CMYK(
				(sd_noise(vec_scale(mat_mulv(HT_MAT_M, pc), 0.8f * dot_size_inv))) * 0.25f * dot_size,
				(sd_noise(vec_scale(mat_mulv(HT_MAT_C, pm), 0.8f * dot_size_inv))) * 0.25f * dot_size,
				(sd_noise(vec_scale(mat_mulv(HT_MAT_K, py), 0.8f * dot_size_inv))) * 0.25f * dot_size,
				(sd_noise(vec_scale(mat_mulv(HT_MAT_Y, pk), 0.8f * dot_size_inv))) * 0.25f * dot_size
			);

			pc = vec_subs(pc, wobble.c);
			pm = vec_subs(pm, wobble.m);
			py = vec_subs(py, wobble.y);
			pk = vec_subs(pk, wobble.k);

			cmyk size = CMYK(
				sample_pixel(img, VEC(in_w, in_h), VEC(width, height), pc, HT_MAT_C, sf, dot_size).c,
				sample_pixel(img, VEC(in_w, in_h), VEC(width, height), pm, HT_MAT_M, sf, dot_size).m,
				sample_pixel(img, VEC(in_w, in_h), VEC(width, height), py, HT_MAT_Y, sf, dot_size).y,
				sample_pixel(img, VEC(in_w, in_h), VEC(width, height), pk, HT_MAT_K, sf, dot_size).k
			);
			// ----

			pc = vec_scale(pc, dot_size_inv);
			pm = vec_scale(pm, dot_size_inv);
			py = vec_scale(py, dot_size_inv);
			pk = vec_scale(pk, dot_size_inv);

			float noise = (sd_noise(vec_scale(pc, 7)) * dot_size) * 0.15;

			sdf sd_c = sd_halftone(pc, HT_MAT_C, size.c) * dot_size + noise;
			sdf sd_m = sd_halftone(pm, HT_MAT_M, size.m) * dot_size + noise;
			sdf sd_y = sd_halftone(py, HT_MAT_Y, size.y) * dot_size + noise;
			sdf sd_k = sd_halftone(pk, HT_MAT_K, size.k) * dot_size + noise;

			cmyk color = {0};

			color = cmyk_mix(color, CMYK(sd_clamp(sd_c, smear) * 0.95, 0, 0, 0));
			color = cmyk_mix(color, CMYK(0, sd_clamp(sd_m, smear) * 0.95, 0, 0));
			color = cmyk_mix(color, CMYK(0, 0, sd_clamp(sd_y, smear) * 0.95, 0));
			color = cmyk_mix(color, CMYK(0, 0, 0, sd_clamp(sd_k, smear) * 0.95));

			image[x + y * width] = cmyk_to_rgb(color);
		}
	}

	time_end = clock();
	printf("%f seconds\n", ((double)time_end - (double)time_start)* 1.0e-6);

	ppm_write("halftone.ppm", width, height, image, image_size);

	free(image);
	free(img);

	return EXIT_SUCCESS;
}
