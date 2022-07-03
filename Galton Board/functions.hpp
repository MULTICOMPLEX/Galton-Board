
#pragma once
#include <future>
#include <numbers>
#include <span>
#include "MULTICOMPLEX.hpp"
//#include "coroutine.hpp"
#include "matplotlib.hpp"
#include "number_theory.hpp"

plot_matplotlib plot;

auto Galton_Classic = []<typename L, typename T>
	requires std::same_as<L, uint64_t>&&
std::integral<T>
(
	const L& balls,
	const T& Board_SIZE,
	auto& galton_arr)
{
	const bool rw = false;

	mxws <uint32_t>RNG;
	double random_walk = 0;
	T k;

	const double mean = Board_SIZE / 2, stddev = 1;

	cxx::ziggurat_normal_distribution<double> normalz(mean, (Board_SIZE / 12) * stddev);

	for (L i = 0; i < balls; i++, random_walk = 0) {

		if (rw) {
			for (T j = 0; j < Board_SIZE; j++)
				random_walk += RNG(1.);

			k = T((random_walk - mean) / sqrt(12. / Board_SIZE) * stddev + mean);
		}

		else {

			k = T(normalz(RNG));

			//The 1D board
			if (k < Board_SIZE) galton_arr[k]++;
		}
	}
};

template <typename R, typename A, typename I, typename B>
	requires std::integral<I>&&
std::same_as<A, uint64_t>
std::tuple<R, I> Galton(
	const A& trials,
	const I& Board_SIZE,
	const I& N_cycles,
	auto& galton_arr,
	B& probability_wave)
{
	mxws <uint32_t>RNG;

	std::tuple<R, I> tuple;

	if (probability_wave)
		tuple = RNG.Probability_Wave<R>(Board_SIZE, galton_arr, N_cycles, trials);

	else {
		 Galton_Classic(trials, Board_SIZE, galton_arr);
		tuple = std::make_tuple(0., Board_SIZE);
	}

	return tuple;
}

template <typename T>
T logint
(
	const T n)
{
	auto lg = 0;
	while ((T(1) << lg) < n)
		lg++;
	return lg;
}

template <typename T>
T rev
(
	T x,
	T lgn
) {
	T res = 0;
	while (lgn--) {
		res = 2 * res + (x & 1);
		x /= 2;
	}
	return res;
}

template <typename T>
std::vector<T> FFT
(
	const std::vector<T>& v)
{
	typedef T Complex;
	const Complex J(0, 1);

	auto n = v.size();
	auto lgn = logint(n);
	assert((n & (n - 1)) == 0);
	std::vector<T> perm(n);

	for (size_t i = 0; auto & d : perm)
		d = v[rev(i++, lgn)];

	for (auto s = 1; s <= lgn; s++) {
		auto m = (1 << s);
		T wm = exp(-2 * std::numbers::pi * J / m);
		for (auto k = 0; k < n; k += m) {
			T w = 1;
			for (auto j = 0; j < m / 2; j++) {
				T t = w * perm[k + j + m / 2];
				T u = perm[k + j];
				perm[k + j] = u + t;
				perm[k + j + m / 2] = u - t;
				w *= wm;
			}
		}
	}
	return perm;
}

template <typename T, typename Y>
void doDFT(std::vector<T>& in, std::vector<Y>& out)
{
	auto N = in.size();
	MX0	z(0, -2 * pi / N);
	MX0 W(exp(z)), Wk(1);

	for (auto& x : out)
	{
		MX0	Wkl(1);
		x = 0;

		for (auto& y : in)
		{
			x += y * Wkl;
			Wkl *= Wk;
		}
		Wk *= W;
	}
}

template <typename T>
void normalize_vector
(
	std::vector<T>& v,
	const double a,
	const double b)
{
	auto k = std::ranges::minmax_element(v);
	auto min = *k.min;
	auto max = *k.max;

	auto normalize = [&](auto& n) {n = a + (n - min) * (b - a) / (max - min); };

	std::ranges::for_each(v, normalize);
}

template <typename T>
void normalize_vector
(
	std::span<T>& v,
	const double a,
	const double b)
{
	auto k = std::ranges::minmax_element(v);
	auto min = *k.min;
	auto max = *k.max;

	auto normalize = [&](auto& n) {n = a + (n - min) * (b - a) / (max - min); };

	std::ranges::for_each(v, normalize);
}

std::vector<std::string> ones{ "","one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };
std::vector<std::string> teens{ "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen","sixteen", "seventeen", "eighteen", "nineteen" };
std::vector<std::string> tens{ "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };

template <typename T>
std::string nameForNumber
(
	const T number)
{
	if (number < 10) {
		return ones[number];
	}
	else if (number < 20) {
		return teens[number - 10];
	}
	else if (number < 100) {
		return tens[number / 10] + ((number % 10 != 0) ? " " + nameForNumber(number % 10) : "");
	}
	else if (number < 1000) {
		return nameForNumber(number / 100) + " hundred" + ((number % 100 != 0) ? " " + nameForNumber(number % 100) : "");
	}
	else if (number < 1000000) {
		return nameForNumber(number / 1000) + " thousand" + ((number % 1000 != 0) ? " " + nameForNumber(number % 1000) : "");
	}
	else if (number < 1000000000) {
		return nameForNumber(number / 1000000) + " million" + ((number % 1000000 != 0) ? " " + nameForNumber(number % 1000000) : "");
	}
	else if (number < 1000000000000) {
		return nameForNumber(number / 1000000000) + " billion" + ((number % 1000000000 != 0) ? " " + nameForNumber(number % 1000000000) : "");
	}
	return "error";
}

template <typename T>
void Sine_Wave_function
(
	std::vector<T>& Y,
	auto freq
) {

	auto k = 1. / (Y.size() / 360.);
	const T rads = k * std::numbers::pi / 180;

	for (auto i = 0; i < Y.size(); i++)
		Y[i] = std::sin(freq * i * rads);
}

template <typename T>
void Cosine_Wave_function
(
	std::vector<T>& Y,
	auto freq
) {

	auto k = 1. / (Y.size() / 360.);
	const T rads = k * std::numbers::pi / 180;

	for (auto i = 0; i < Y.size(); i++)
		Y[i] = std::cos(freq * i * rads);
}

template <typename T>
void Hann_function
(
	std::vector<T>& Y
) {

	for (auto i = 0; i < Y.size(); i++)
	{
		double multiplier = 0.5 * (1 - std::cos(2 * std::numbers::pi * i / (Y.size() - 1)));
		Y[i] *= multiplier;
	}
}

int MakeWavFromVector
(
	std::string file_name,
	int sample_rate_arg,
	std::vector<short>& sound
);


const char line[] = "    +------+------+------+------+------+------+------+------+------+------+------+------+------+----->";

template <typename T>
void cout_galton
(
	const T nboxes,
	const std::vector<T>& galton_arr
) {
	unsigned i, j;

	auto max = double(*std::ranges::max_element(galton_arr));

	max /= sizeof(line) - 7;

	std::cout << line << std::endl;

	/* print out the results */
	for (i = 0; i < nboxes; i++) {

		std::cout << std::setw(3) << i << " |";
		for (j = 0; j < ceil(galton_arr[i] / float(max)); j++) {
			/* the '#' indicates the balls */
			std::cout << "#";
		}
		std::cout << " " << galton_arr[i] << std::endl << line << std::endl;
	}
}

template <typename T>
bool findParity(T x)
{
	T y = x ^ (x >> 1);
	y ^= std::rotr(y, 2);
	y ^= std::rotr(y, 4);
	y ^= std::rotr(y, 8);
	y ^= std::rotr(x, 32);
	y ^= std::size_t(y) >> 32;

	// Rightmost bit of y holds the parity value
	// if (y&1) is 1 then parity is odd else even
	if (y & 1)
		return 1;
	return 0;
}

template <typename T>
void null_offset_vector(std::vector<T>& v)
{
	T mean = 0;
	for (auto& d : v)
		mean += d;

	mean /= v.size();

	for (auto& d : v)
		d -= mean;
}

inline double to_int(double d)
{
	d += 6755399441055744.0;
	return reinterpret_cast<int&>(d);
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}

template <typename T>
T inline fast_mod(const T& input, const T& ceil)
{
	// apply the modulo operator only when needed
	// (i.e. when the input is greater than the ceiling)
	return input >= ceil ? input % ceil : input;
	// NB: the assumption here is that the numbers are positive
}

std::string utf8_encode(std::u8string const& s)
{
	return (const char*)(s.c_str());
}

std::ostream& operator <<
(
	std::ostream& o,
	std::u8string& s
	)
{
	o << (const char*)(s.c_str());
	return o;
}

#ifdef _WIN32

#include <Windows.h>

class UTF8CodePage {
public:
	UTF8CodePage() : m_old_code_page(GetConsoleOutputCP()) {
		SetConsoleOutputCP(CP_UTF8);
	}
	~UTF8CodePage() { SetConsoleOutputCP(m_old_code_page); }

private:
	UINT m_old_code_page;
};

UTF8CodePage use_utf8;

#endif

template <class NumberFormat, std::size_t DFT_Length>
class SlidingDFT
{
private:
	/// Are the frequency domain values valid? (i.e. have at elast DFT_Length data
	/// points been seen?)
	bool data_valid = false;

	/// Time domain samples are stored in this circular buffer.
	std::array<NumberFormat, DFT_Length> x = { 0 };

	/// Index of the next item in the buffer to be used. Equivalently, the number
	/// of samples that have been seen so far modulo DFT_Length.
	size_t x_index = 0;

	/// Twiddle factors for the update algorithm
	std::array <MX0, DFT_Length> twiddle;

public:
	/// Frequency domain values 
	std::array <MX0, DFT_Length> dft;

	virtual ~SlidingDFT() = default;

	/// Constructor
	SlidingDFT()
	{
		const MX0 j(0, 1);
		const NumberFormat N = DFT_Length;

		// Compute the twiddle factors, and zero the x 
		for (size_t k = 0; k < DFT_Length; k++) {
			NumberFormat factor = (NumberFormat)(2 * std::numbers::pi) * k / N;
			twiddle[k] = exp(j * factor);
		}
	}

	/// Determine whether the output data is valid
	bool is_data_valid()
	{
		return data_valid;
	}

	/// Update the calculation with a new sample
	/// Returns true if the data are valid (because enough samples have been
	/// presented), or false if the data are invalid.
	bool update(NumberFormat new_x)
	{
		// Update the storage of the time domain values
		const NumberFormat old_x = x[x_index];
		x[x_index] = new_x;

		// Update the DFT
		for (size_t k = 0; k < DFT_Length; k++) {
			dft[k] = twiddle[k] * (dft[k] - old_x + new_x);
		}

		// Increment the counter
		x_index++;
		if (x_index >= DFT_Length) {
			data_valid = true;
			x_index = 0;
		}

		// Done.
		return data_valid;
	}
};

template <typename T>
void slidingDFT_driver(
	const T N,
	auto& cx,
	auto& Y)
{
	SlidingDFT<double, 2037> dft;

	for (size_t i = 0; i < N; i++) {
		dft.update(Y[i]);
		if (dft.is_data_valid()) {
			for (size_t j = 0; j < N; j++)
				cx[j] = dft.dft[j];
		}
	}
}

struct Color {
	union {
		struct { unsigned char b, g, r, a; };
		unsigned char bytes[4];
		unsigned int ref;
	};
	Color(unsigned int ref = 0) : ref(ref) {}
};

class Surface {
	int width, height;
	std::vector<Color> pixels;
public:
	void Fill(Color color) { std::fill(pixels.begin(), pixels.end(), color); }
	void HLine(int left, int y, int len, Color color) {
		if (y < 0 || y >= height || left >= width) { return; }
		if (left < 0) { len += left; left = 0; }
		if (left + len > width) { len = width - left; }
		int offset = y * width + left;
		std::fill(pixels.begin() + offset, pixels.begin() + offset + len, color);
	}
	void RectFill(int x, int y, int w, int h, Color color) {
		for (int i = 0; i < h; ++i) { HLine(x, y + i, w, color); }
	}
	Surface(int width, int height) :
		width(width),
		height(height),
		pixels(width* height, Color())
	{}
	template <typename I>       Color& operator () (const I& x, const I& y) { return pixels[y * width + x]; }
	template <typename I> const Color& operator () (const I& x, const I& y) const { return pixels[y * width + x]; }

	class Writer {
		std::ofstream ofs;
	public:
		Writer(const char* filename) : ofs(filename, std::ios_base::out | std::ios_base::binary) {}
		void operator () (const void* pbuf, int size) { ofs.write(static_cast<const char*>(pbuf), size); }
		template <typename T> void operator () (const T& obj) { operator () (&obj, sizeof(obj)); }
	};

	struct BIH {
		unsigned int   sz;
		int            width, height;
		unsigned short planes;
		short          bits;
		unsigned int   compress, szimage;
		int            xppm, yppm;
		unsigned int   clrused, clrimp;
	};

	void Save(const char* filename) const {
		Writer w(filename);;
		w("BM", 2);
		BIH bih = { sizeof(bih) };
		bih.width = width;
		bih.height = -height;
		bih.planes = 1;
		bih.bits = 32;
		const unsigned int headersize = sizeof(bih) + 14;
		const int szbuf = int(sizeof(Color) * pixels.size());
		const unsigned int filesize = static_cast<unsigned int>(headersize + szbuf);
		w(filesize);
		const unsigned short z = 0;
		w(z);
		w(z);
		w(headersize);
		w(bih);
		w(pixels.data(), szbuf);
	}
};

void Circle_Squares_fractal(int width = 1024, int height = 1024)
{
	Surface surf(width, height);
	Color color;

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			auto x = (i * i + j * j) / 4;
			color.r = color.g = color.b = 255 - x;
			surf.RectFill(i, j, 1, 1, color);
		}
	}
	surf.Save("Circle_Squares_fractal.bmp");
}