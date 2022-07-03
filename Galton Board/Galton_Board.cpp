
#include "functions.hpp"

/*1D Galton Board simulation with normal and sinusoidal distribution*/

int main(int argc, char** argv)
{

	std::setlocale(LC_ALL, "en_US.utf8");

	typedef unsigned I;
	typedef double R;
	typedef bool B;

	/***************SETTINGS*****************/

	std::uint64_t N_Trials = 50000000;

	I N_cycles = 8; //== Threads 

	I N_Bins = 2048;

	//Sinusoidal Distribution
	B probability_wave = true;
	//Raw distribution
	B raw = false;
	//Enable Fourier transform
	B dft = true;
	//Enable sound to file (wav format)
	B wav = false;

	//Console output
	const B cout_gal = false;

	/***************SETTINGS*****************/

	if (wav) {
		N_Trials = 100000000;
		N_Bins = 300000;
		N_cycles = 1000;
		dft = false;
	}

	I Board_SIZE = I(round(N_Bins / R(N_cycles)));

	/* get cmd args */
	if (argc < 7) {
		std::cout <<
			"Usage: Galton_Board [N]Trials(Balls) [N]Cycles [1-0]Probability-Wave\
 [1-0]Raw [1-0]DFT [1-0]Sound-wav" <<
			std::endl << std::endl;
	}

	else {
		N_Trials = atoi(argv[1]);
		N_cycles = atoi(argv[2]);
		probability_wave = atoi(argv[3]);
		raw = atoi(argv[4]);
		dft = atoi(argv[5]);
		wav = atoi(argv[6]);
		Board_SIZE = I(round(N_Bins / R(N_cycles)));
	}

	std::u8string title = u8" Probability Wave Ψ";
	if (!probability_wave) title = u8" Galton Board";
	std::cout << title << std::endl << std::endl;

	auto N_Hthreads = std::thread::hardware_concurrency();
	std::cout << " " << N_Hthreads << " concurrent cycles (threads) are supported."
		<< std::endl << std::endl;

	if (probability_wave) {
		std::cout << " Trials        " << nameForNumber(N_Trials) << " (" << N_Trials << ")"
			<< std::endl;
		std::cout << " Cycles        " << N_cycles << std::endl;
		std::cout << " Board SIZE    " << Board_SIZE << "[Boxes]" << std::endl;
	}

	else {
		Board_SIZE = 2000;
		N_cycles = 1;
		std::cout << " Trials         " << nameForNumber(N_Trials) << " (" << N_Trials << ")"
			<< std::endl;
	}

	std::tuple<R, I> tuple;

	std::vector < std::future <decltype(tuple)>> vecOfThreads;

	std::vector<std::vector<I>> galton_arr(N_cycles, std::vector<I>(Board_SIZE));

	auto begin = std::chrono::high_resolution_clock::now();

	for (I i = 0; i < N_cycles; i++)
		vecOfThreads.push_back(std::async([&, i] {
		return Galton<R>(N_Trials, Board_SIZE, N_cycles, galton_arr[i], probability_wave); }));

	for (auto& th : vecOfThreads)
		tuple = th.get();

	auto end = std::chrono::high_resolution_clock::now();

	I Board_size = std::get<I>(tuple);

	if (probability_wave)
	{
		std::cout << std::endl << " RNG range     " << std::get<R>(tuple) << "[Boxes]" << std::endl;
	}

	std::cout << " Board size    " << Board_size << "[Boxes]" << std::endl;

	std::cout << std::endl << " Duration Ball "
		<< std::chrono::nanoseconds(end - begin).count() / N_Trials
		<< "[ns]" << std::endl << std::endl;

	if (Board_SIZE <= 256 && cout_gal)
		cout_galton(Board_SIZE, galton_arr[0]);

	std::vector<R> X, Y, Y_buf;

	if (!probability_wave) {

		for (auto i = 0; auto & k : std::span(galton_arr.front())) {
			X.push_back(i++);
			Y.push_back(R(k));
		}

		plot.plot_somedata(X, Y, "", "Binomial-Normal Distribution", "blue");

		std::string str = "Number of Balls : ";
		str += nameForNumber(N_Trials);

		auto max = *std::ranges::max_element(Y);

		plot.text(0, max / 3, str, "green", 11);

		plot.set_xlabel("Boxes");
		plot.grid_on();
		plot.set_title(utf8_encode(title));
		plot.show();
	}

	else {

		for (auto& i : galton_arr)
			std::ranges::transform(i, std::back_inserter(Y),
				[](auto& c) {return c; });

		if (!raw) {
			normalize_vector(Y, 1., -1.);
			null_offset_vector(Y);
		}

		Y_buf = Y;

		B pow2 = ((Y.size() & (Y.size() - 1)) == 0);

		if (wav)
			pow2 = false;

		std::vector<MX0> cx;

		if (((Y.size() <= 1000000) || pow2) && !raw && dft)
		{
			if (pow2) {
				std::ranges::transform(Y, std::back_inserter(cx),
					[](auto& c) {return c; });
				cx = FFT(cx);
			}

			else {
				cx.resize(Y.size());
				doDFT(Y, cx);
			}

			X.clear();
			Y.clear();

			for (auto i = 0.; auto & d : std::span(cx).subspan(0, cx.size() / 2))
			{
				X.push_back(i++);
				Y.push_back(20 * std::log10(std::sqrt(d.norm()) / (cx.size() / 2.) + .001));
			}

			plot.set_ylabel("dB");
			plot.plot_somedata(X, Y, "", "fourier transform", "red");

			R rms = 0;
			for (auto& d : std::span(cx).subspan(0, cx.size() / 2))
				rms += std::sqrt(d.norm());

			rms *= 1. / (cx.size()) * 2;

			std::cout << " RMS	       " << rms << "[dB]" << std::endl << std::endl;

			std::string str = "RMS= ";
			str += std::to_string(rms);
			str += "    RNG range= ";

			auto rng_range = std::get<R>(tuple);
			str += to_string_with_precision(rng_range, 3);
			plot.text(210, -23, str, "green", 11);

			if (Board_SIZE > Board_size) {
				str = "Shrunken to ";
				plot.text(210, -13, str + std::to_string(Board_size), "purple", 11);
				str = "Factor= ";
				plot.text(210, -18, str, "orange", 11);
				str += to_string_with_precision(R(Board_SIZE) / Board_size, 3);
			}

			else if (Board_SIZE < Board_size) {
				str = "Grown to ";
				plot.text(210, -13, str + std::to_string(Board_size), "purple", 11);
				str = "Factor= ";
				plot.text(210, -18, str, "orange", 11);
				str += to_string_with_precision(Board_size / R(Board_SIZE), 3);
			}

			else {
				str = "Size stayed the same= ";
				plot.text(210, -13, str + std::to_string(Board_size), "purple", 11);
				str = "Factor= 1";
			}

			plot.text(210, -18, str, "orange", 11);

			str = "Number of cycles= ";
			str += std::to_string(N_cycles);
			str += ", Board size= ";
			plot.text(210, -8, str + std::to_string(Y_buf.size() / N_cycles),
				"blue", 11);

			X.clear();

			for (auto i = 0.; i < Y_buf.size() / 2.; i += 0.5)
				X.push_back(i);

			plot.line(Y.size() - R(Board_size / 2),
				R(Y.size()), -2, -2, "purple", 4, "solid");
		}

		if (!wav)
		{
			if (!dft || raw) {
				X.clear();
				for (auto i = 0; i < Y_buf.size(); i++)
					X.push_back(i);
			}

			std::string str = "number of trials: ";

			str += nameForNumber(N_Trials);
			str += " x ";
			str += std::to_string(N_cycles);
			plot.plot_somedata(X, Y_buf, "", str, "blue");

			plot.set_title(utf8_encode(title));
			if (!raw)plot.set_xlabel("Bins / 2");
			else plot.set_xlabel("Bins");
			plot.grid_on();
			plot.show();
		}

		else {

			std::vector<short> sound;
			sound.reserve(Y_buf.size());

			for (auto& d : Y_buf)
				sound.push_back(short((d * 2500.)));

			MakeWavFromVector("Galton.wav", 44100, sound);

			if (X.size() <= 20480) {

				X.clear();

				for (auto i = 0.; i < Y_buf.size(); i++)
					X.push_back(i);

				plot.plot_somedata(X, Y_buf, "", "Wav", "blue");

				plot.set_title("Galton.wav");
				plot.show();
			}
		}
	}

	return 0;
}

