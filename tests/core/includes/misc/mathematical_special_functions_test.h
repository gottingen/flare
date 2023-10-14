// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <fstream>
#include <doctest.h>
#include <flare/core.h>

namespace Test {

    struct TestLargeArgTag {
    };
    struct TestRealErfcxTag {
    };

    template<class ExecSpace>
    struct TestExponentialIntergral1Function {
        using TensorType = flare::Tensor<double *, ExecSpace>;
        using HostTensorType = flare::Tensor<double *, flare::HostSpace>;

        TensorType d_x, d_expint;
        typename TensorType::HostMirror h_x, h_expint;
        HostTensorType h_ref;

        void testit() {
            using flare::fabs;
            using flare::experimental::infinity;

            d_x = TensorType("d_x", 15);
            d_expint = TensorType("d_expint", 15);
            h_x = flare::create_mirror_tensor(d_x);
            h_expint = flare::create_mirror_tensor(d_expint);
            h_ref = HostTensorType("h_ref", 15);

            // Generate test inputs
            h_x(0) = -0.2;
            h_x(1) = 0.0;
            h_x(2) = 0.2;
            h_x(3) = 0.8;
            h_x(4) = 1.6;
            h_x(5) = 5.1;
            h_x(6) = 0.01;
            h_x(7) = 0.001;
            h_x(8) = 1.0;
            h_x(9) = 1.001;
            h_x(10) = 1.01;
            h_x(11) = 1.1;
            h_x(12) = 7.2;
            h_x(13) = 10.3;
            h_x(14) = 15.4;
            flare::deep_copy(d_x, h_x);

            // Call exponential integral function
            flare::parallel_for(flare::RangePolicy<ExecSpace>(0, 15), *this);
            flare::fence();

            flare::deep_copy(h_expint, d_expint);

            // Reference values computed with Octave
            h_ref(0) = -infinity<double>::value;  // x(0)=-0.2
            h_ref(1) = infinity<double>::value;   // x(1)= 0.0
            h_ref(2) = 1.222650544183893e+00;     // x(2) =0.2
            h_ref(3) = 3.105965785455429e-01;     // x(3) =0.8
            h_ref(4) = 8.630833369753976e-02;     // x(4) =1.6
            h_ref(5) = 1.021300107861738e-03;     // x(5) =5.1
            h_ref(6) = 4.037929576538113e+00;     // x(6) =0.01
            h_ref(7) = 6.331539364136149e+00;     // x(7) =0.001
            h_ref(8) = 2.193839343955205e-01;     // x(8) =1.0
            h_ref(9) = 2.190164225274689e-01;     // x(9) =1.001
            h_ref(10) = 2.157416237944899e-01;     // x(10)=1.01
            h_ref(11) = 1.859909045360401e-01;     // x(11)=1.1
            h_ref(12) = 9.218811688716196e-05;     // x(12)=7.2
            h_ref(13) = 2.996734771597901e-06;     // x(13)=10.3
            h_ref(14) = 1.254522935050609e-08;     // x(14)=15.4

            REQUIRE_EQ(h_ref(0), h_expint(0));
            REQUIRE_EQ(h_ref(1), h_expint(1));
            for (int i = 2; i < 15; i++) {
                REQUIRE_LE(std::abs(h_expint(i) - h_ref(i)), std::abs(h_ref(i)) * 1e-15);
            }
        }

        FLARE_INLINE_FUNCTION
        void operator()(const int &i) const {
            d_expint(i) = flare::experimental::expint1(d_x(i));
        }
    };

    template<class ExecSpace>
    struct TestComplexErrorFunction {
        using TensorType = flare::Tensor<flare::complex<double> *, ExecSpace>;
        using HostTensorType =
                flare::Tensor<flare::complex<double> *, flare::HostSpace>;
        using DblTensorType = flare::Tensor<double *, ExecSpace>;
        using DblHostTensorType = flare::Tensor<double *, flare::HostSpace>;

        TensorType d_z, d_erf, d_erfcx;
        typename TensorType::HostMirror h_z, h_erf, h_erfcx;
        HostTensorType h_ref_erf, h_ref_erfcx;

        DblTensorType d_x, d_erfcx_dbl;
        typename DblTensorType::HostMirror h_x, h_erfcx_dbl;
        DblHostTensorType h_ref_erfcx_dbl;

        void testit() {
            using flare::experimental::infinity;

            d_z = TensorType("d_z", 52);
            d_erf = TensorType("d_erf", 52);
            d_erfcx = TensorType("d_erfcx", 52);
            h_z = flare::create_mirror_tensor(d_z);
            h_erf = flare::create_mirror_tensor(d_erf);
            h_erfcx = flare::create_mirror_tensor(d_erfcx);
            h_ref_erf = HostTensorType("h_ref_erf", 52);
            h_ref_erfcx = HostTensorType("h_ref_erfcx", 52);

            d_x = DblTensorType("d_x", 6);
            d_erfcx_dbl = DblTensorType("d_erfcx_dbl", 6);
            h_x = flare::create_mirror_tensor(d_x);
            h_erfcx_dbl = flare::create_mirror_tensor(d_erfcx_dbl);
            h_ref_erfcx_dbl = DblHostTensorType("h_ref_erfcx_dbl", 6);

            // Generate test inputs
            // abs(z)<=2
            h_z(0) = flare::complex<double>(0.0011, 0);
            h_z(1) = flare::complex<double>(-0.0011, 0);
            h_z(2) = flare::complex<double>(1.4567, 0);
            h_z(3) = flare::complex<double>(-1.4567, 0);
            h_z(4) = flare::complex<double>(0, 0.0011);
            h_z(5) = flare::complex<double>(0, -0.0011);
            h_z(6) = flare::complex<double>(0, 1.4567);
            h_z(7) = flare::complex<double>(0, -1.4567);
            h_z(8) = flare::complex<double>(1.4567, 0.0011);
            h_z(9) = flare::complex<double>(1.4567, -0.0011);
            h_z(10) = flare::complex<double>(-1.4567, 0.0011);
            h_z(11) = flare::complex<double>(-1.4567, -0.0011);
            h_z(12) = flare::complex<double>(1.4567, 0.5942);
            h_z(13) = flare::complex<double>(1.4567, -0.5942);
            h_z(14) = flare::complex<double>(-1.4567, 0.5942);
            h_z(15) = flare::complex<double>(-1.4567, -0.5942);
            h_z(16) = flare::complex<double>(0.0011, 0.5942);
            h_z(17) = flare::complex<double>(0.0011, -0.5942);
            h_z(18) = flare::complex<double>(-0.0011, 0.5942);
            h_z(19) = flare::complex<double>(-0.0011, -0.5942);
            h_z(20) = flare::complex<double>(0.0011, 0.0051);
            h_z(21) = flare::complex<double>(0.0011, -0.0051);
            h_z(22) = flare::complex<double>(-0.0011, 0.0051);
            h_z(23) = flare::complex<double>(-0.0011, -0.0051);
            // abs(z)>2.0 and x>1
            h_z(24) = flare::complex<double>(3.5, 0.0011);
            h_z(25) = flare::complex<double>(3.5, -0.0011);
            h_z(26) = flare::complex<double>(-3.5, 0.0011);
            h_z(27) = flare::complex<double>(-3.5, -0.0011);
            h_z(28) = flare::complex<double>(3.5, 9.7);
            h_z(29) = flare::complex<double>(3.5, -9.7);
            h_z(30) = flare::complex<double>(-3.5, 9.7);
            h_z(31) = flare::complex<double>(-3.5, -9.7);
            h_z(32) = flare::complex<double>(18.9, 9.7);
            h_z(33) = flare::complex<double>(18.9, -9.7);
            h_z(34) = flare::complex<double>(-18.9, 9.7);
            h_z(35) = flare::complex<double>(-18.9, -9.7);
            // abs(z)>2.0 and 0<=x<=1 and abs(y)<6
            h_z(36) = flare::complex<double>(0.85, 3.5);
            h_z(37) = flare::complex<double>(0.85, -3.5);
            h_z(38) = flare::complex<double>(-0.85, 3.5);
            h_z(39) = flare::complex<double>(-0.85, -3.5);
            h_z(40) = flare::complex<double>(0.0011, 3.5);
            h_z(41) = flare::complex<double>(0.0011, -3.5);
            h_z(42) = flare::complex<double>(-0.0011, 3.5);
            h_z(43) = flare::complex<double>(-0.0011, -3.5);
            // abs(z)>2.0 and 0<=x<=1 and abs(y)>=6
            h_z(44) = flare::complex<double>(0.85, 7.5);
            h_z(45) = flare::complex<double>(0.85, -7.5);
            h_z(46) = flare::complex<double>(-0.85, 7.5);
            h_z(47) = flare::complex<double>(-0.85, -7.5);
            h_z(48) = flare::complex<double>(0.85, 19.7);
            h_z(49) = flare::complex<double>(0.85, -19.7);
            h_z(50) = flare::complex<double>(-0.85, 19.7);
            h_z(51) = flare::complex<double>(-0.85, -19.7);

            h_x(0) = -infinity<double>::value;
            h_x(1) = -1.2;
            h_x(2) = 0.0;
            h_x(3) = 1.2;
            h_x(4) = 10.5;
            h_x(5) = infinity<double>::value;

            flare::deep_copy(d_z, h_z);
            flare::deep_copy(d_x, h_x);

            // Call erf and erfcx functions
            flare::parallel_for(flare::RangePolicy<ExecSpace>(0, 52), *this);
            flare::fence();

            flare::parallel_for(flare::RangePolicy<ExecSpace, TestRealErfcxTag>(0, 1),
                                *this);
            flare::fence();

            flare::deep_copy(h_erf, d_erf);
            flare::deep_copy(h_erfcx, d_erfcx);
            flare::deep_copy(h_erfcx_dbl, d_erfcx_dbl);

            // Reference values computed with Octave
            h_ref_erf(0) = flare::complex<double>(0.001241216583181022, 0);
            h_ref_erf(1) = flare::complex<double>(-0.001241216583181022, 0);
            h_ref_erf(2) = flare::complex<double>(0.9606095744865353, 0);
            h_ref_erf(3) = flare::complex<double>(-0.9606095744865353, 0);
            h_ref_erf(4) = flare::complex<double>(0, 0.001241217584429469);
            h_ref_erf(5) = flare::complex<double>(0, -0.001241217584429469);
            h_ref_erf(6) = flare::complex<double>(0, 4.149756424218223);
            h_ref_erf(7) = flare::complex<double>(0, -4.149756424218223);
            h_ref_erf(8) =
                    flare::complex<double>(0.960609812745064, 0.0001486911741082233);
            h_ref_erf(9) =
                    flare::complex<double>(0.960609812745064, -0.0001486911741082233);
            h_ref_erf(10) =
                    flare::complex<double>(-0.960609812745064, 0.0001486911741082233);
            h_ref_erf(11) =
                    flare::complex<double>(-0.960609812745064, -0.0001486911741082233);
            h_ref_erf(12) =
                    flare::complex<double>(1.02408827958197, 0.04828570635603527);
            h_ref_erf(13) =
                    flare::complex<double>(1.02408827958197, -0.04828570635603527);
            h_ref_erf(14) =
                    flare::complex<double>(-1.02408827958197, 0.04828570635603527);
            h_ref_erf(15) =
                    flare::complex<double>(-1.02408827958197, -0.04828570635603527);
            h_ref_erf(16) =
                    flare::complex<double>(0.001766791817179109, 0.7585038120712589);
            h_ref_erf(17) =
                    flare::complex<double>(0.001766791817179109, -0.7585038120712589);
            h_ref_erf(18) =
                    flare::complex<double>(-0.001766791817179109, 0.7585038120712589);
            h_ref_erf(19) =
                    flare::complex<double>(-0.001766791817179109, -0.7585038120712589);
            h_ref_erf(20) =
                    flare::complex<double>(0.001241248867618165, 0.005754776682713324);
            h_ref_erf(21) =
                    flare::complex<double>(0.001241248867618165, -0.005754776682713324);
            h_ref_erf(22) =
                    flare::complex<double>(-0.001241248867618165, 0.005754776682713324);
            h_ref_erf(23) =
                    flare::complex<double>(-0.001241248867618165, -0.005754776682713324);
            h_ref_erf(24) =
                    flare::complex<double>(0.9999992569244941, 5.939313159932013e-09);
            h_ref_erf(25) =
                    flare::complex<double>(0.9999992569244941, -5.939313159932013e-09);
            h_ref_erf(26) =
                    flare::complex<double>(-0.9999992569244941, 5.939313159932013e-09);
            h_ref_erf(27) =
                    flare::complex<double>(-0.9999992569244941, -5.939313159932013e-09);
            h_ref_erf(28) =
                    flare::complex<double>(-1.915595842013002e+34, 1.228821279117683e+32);
            h_ref_erf(29) =
                    flare::complex<double>(-1.915595842013002e+34, -1.228821279117683e+32);
            h_ref_erf(30) =
                    flare::complex<double>(1.915595842013002e+34, 1.228821279117683e+32);
            h_ref_erf(31) =
                    flare::complex<double>(1.915595842013002e+34, -1.228821279117683e+32);
            h_ref_erf(32) = flare::complex<double>(1, 5.959897539826596e-117);
            h_ref_erf(33) = flare::complex<double>(1, -5.959897539826596e-117);
            h_ref_erf(34) = flare::complex<double>(-1, 5.959897539826596e-117);
            h_ref_erf(35) = flare::complex<double>(-1, -5.959897539826596e-117);
            h_ref_erf(36) =
                    flare::complex<double>(-9211.077162784413, 13667.93825589455);
            h_ref_erf(37) =
                    flare::complex<double>(-9211.077162784413, -13667.93825589455);
            h_ref_erf(38) =
                    flare::complex<double>(9211.077162784413, 13667.93825589455);
            h_ref_erf(39) =
                    flare::complex<double>(9211.077162784413, -13667.93825589455);
            h_ref_erf(40) = flare::complex<double>(259.38847811225, 35281.28906479814);
            h_ref_erf(41) =
                    flare::complex<double>(259.38847811225, -35281.28906479814);
            h_ref_erf(42) =
                    flare::complex<double>(-259.38847811225, 35281.28906479814);
            h_ref_erf(43) =
                    flare::complex<double>(-259.38847811225, -35281.28906479814);
            h_ref_erf(44) =
                    flare::complex<double>(6.752085728270252e+21, 9.809477366939276e+22);
            h_ref_erf(45) =
                    flare::complex<double>(6.752085728270252e+21, -9.809477366939276e+22);
            h_ref_erf(46) =
                    flare::complex<double>(-6.752085728270252e+21, 9.809477366939276e+22);
            h_ref_erf(47) =
                    flare::complex<double>(-6.752085728270252e+21, -9.809477366939276e+22);
            h_ref_erf(48) =
                    flare::complex<double>(4.37526734926942e+166, -2.16796709605852e+166);
            h_ref_erf(49) =
                    flare::complex<double>(4.37526734926942e+166, 2.16796709605852e+166);
            h_ref_erf(50) =
                    flare::complex<double>(-4.37526734926942e+166, -2.16796709605852e+166);
            h_ref_erf(51) =
                    flare::complex<double>(-4.37526734926942e+166, 2.16796709605852e+166);

            h_ref_erfcx(0) = flare::complex<double>(0.9987599919156778, 0);
            h_ref_erfcx(1) = flare::complex<double>(1.001242428085786, 0);
            h_ref_erfcx(2) = flare::complex<double>(0.3288157848563544, 0);
            h_ref_erfcx(3) = flare::complex<double>(16.36639786516915, 0);
            h_ref_erfcx(4) =
                    flare::complex<double>(0.999998790000732, -0.001241216082557101);
            h_ref_erfcx(5) =
                    flare::complex<double>(0.999998790000732, 0.001241216082557101);
            h_ref_erfcx(6) =
                    flare::complex<double>(0.1197948131677216, -0.4971192955307743);
            h_ref_erfcx(7) =
                    flare::complex<double>(0.1197948131677216, 0.4971192955307743);
            h_ref_erfcx(8) =
                    flare::complex<double>(0.3288156873503045, -0.0001874479383970247);
            h_ref_erfcx(9) =
                    flare::complex<double>(0.3288156873503045, 0.0001874479383970247);
            h_ref_erfcx(10) =
                    flare::complex<double>(16.36629202874158, -0.05369111060785572);
            h_ref_erfcx(11) =
                    flare::complex<double>(16.36629202874158, 0.05369111060785572);
            h_ref_erfcx(12) =
                    flare::complex<double>(0.3020886508118801, -0.09424097887578842);
            h_ref_erfcx(13) =
                    flare::complex<double>(0.3020886508118801, 0.09424097887578842);
            h_ref_erfcx(14) =
                    flare::complex<double>(-2.174707722732267, -11.67259764091796);
            h_ref_erfcx(15) =
                    flare::complex<double>(-2.174707722732267, 11.67259764091796);
            h_ref_erfcx(16) =
                    flare::complex<double>(0.7019810779371267, -0.5319516793968513);
            h_ref_erfcx(17) =
                    flare::complex<double>(0.7019810779371267, 0.5319516793968513);
            h_ref_erfcx(18) =
                    flare::complex<double>(0.7030703366403597, -0.5337884198542978);
            h_ref_erfcx(19) =
                    flare::complex<double>(0.7030703366403597, 0.5337884198542978);
            h_ref_erfcx(20) =
                    flare::complex<double>(0.9987340467266177, -0.005743428170378673);
            h_ref_erfcx(21) =
                    flare::complex<double>(0.9987340467266177, 0.005743428170378673);
            h_ref_erfcx(22) =
                    flare::complex<double>(1.001216353762532, -0.005765867613873103);
            h_ref_erfcx(23) =
                    flare::complex<double>(1.001216353762532, 0.005765867613873103);
            h_ref_erfcx(24) =
                    flare::complex<double>(0.1552936427089241, -4.545593205871305e-05);
            h_ref_erfcx(25) =
                    flare::complex<double>(0.1552936427089241, 4.545593205871305e-05);
            h_ref_erfcx(26) =
                    flare::complex<double>(417949.5262869648, -3218.276197742372);
            h_ref_erfcx(27) =
                    flare::complex<double>(417949.5262869648, 3218.276197742372);
            h_ref_erfcx(28) =
                    flare::complex<double>(0.01879467905925653, -0.0515934271478583);
            h_ref_erfcx(29) =
                    flare::complex<double>(0.01879467905925653, 0.0515934271478583);
            h_ref_erfcx(30) =
                    flare::complex<double>(-0.01879467905925653, -0.0515934271478583);
            h_ref_erfcx(31) =
                    flare::complex<double>(-0.01879467905925653, 0.0515934271478583);
            h_ref_erfcx(32) =
                    flare::complex<double>(0.02362328821805, -0.01209735551897239);
            h_ref_erfcx(33) =
                    flare::complex<double>(0.02362328821805, 0.01209735551897239);
            h_ref_erfcx(34) = flare::complex<double>(-2.304726099084567e+114,
                                                     -2.942443198107089e+114);
            h_ref_erfcx(35) = flare::complex<double>(-2.304726099084567e+114,
                                                     2.942443198107089e+114);
            h_ref_erfcx(36) =
                    flare::complex<double>(0.04174017523145063, -0.1569865319886248);
            h_ref_erfcx(37) =
                    flare::complex<double>(0.04174017523145063, 0.1569865319886248);
            h_ref_erfcx(38) =
                    flare::complex<double>(-0.04172154858670504, -0.156980085534407);
            h_ref_erfcx(39) =
                    flare::complex<double>(-0.04172154858670504, 0.156980085534407);
            h_ref_erfcx(40) =
                    flare::complex<double>(6.355803055239174e-05, -0.1688298297427782);
            h_ref_erfcx(41) =
                    flare::complex<double>(6.355803055239174e-05, 0.1688298297427782);
            h_ref_erfcx(42) =
                    flare::complex<double>(-5.398806789669434e-05, -0.168829903432947);
            h_ref_erfcx(43) =
                    flare::complex<double>(-5.398806789669434e-05, 0.168829903432947);
            h_ref_erfcx(44) =
                    flare::complex<double>(0.008645103282302355, -0.07490521021566741);
            h_ref_erfcx(45) =
                    flare::complex<double>(0.008645103282302355, 0.07490521021566741);
            h_ref_erfcx(46) =
                    flare::complex<double>(-0.008645103282302355, -0.07490521021566741);
            h_ref_erfcx(47) =
                    flare::complex<double>(-0.008645103282302355, 0.07490521021566741);
            h_ref_erfcx(48) =
                    flare::complex<double>(0.001238176693606428, -0.02862247416909219);
            h_ref_erfcx(49) =
                    flare::complex<double>(0.001238176693606428, 0.02862247416909219);
            h_ref_erfcx(50) =
                    flare::complex<double>(-0.001238176693606428, -0.02862247416909219);
            h_ref_erfcx(51) =
                    flare::complex<double>(-0.001238176693606428, 0.02862247416909219);

            h_ref_erfcx_dbl(0) = infinity<double>::value;
            h_ref_erfcx_dbl(1) = 8.062854217063865e+00;
            h_ref_erfcx_dbl(2) = 1.0;
            h_ref_erfcx_dbl(3) = 3.785374169292397e-01;
            h_ref_erfcx_dbl(4) = 5.349189974656411e-02;
            h_ref_erfcx_dbl(5) = 0.0;

            for (int i = 0; i < 52; i++) {
                REQUIRE_LE(flare::abs(h_erf(i) - h_ref_erf(i)),
                           flare::abs(h_ref_erf(i)) * 1e-13);
            }

            for (int i = 0; i < 52; i++) {
                REQUIRE_LE(flare::abs(h_erfcx(i) - h_ref_erfcx(i)),
                           flare::abs(h_ref_erfcx(i)) * 1e-13);
            }

            REQUIRE_EQ(h_erfcx_dbl(0), h_ref_erfcx_dbl(0));
            REQUIRE_EQ(h_erfcx_dbl(5), h_ref_erfcx_dbl(5));
            for (int i = 1; i < 5; i++) {
                REQUIRE_LE(std::abs(h_erfcx_dbl(i) - h_ref_erfcx_dbl(i)),
                           std::abs(h_ref_erfcx_dbl(i)) * 1e-13);
            }
        }

        FLARE_INLINE_FUNCTION
        void operator()(const int &i) const {
            d_erf(i) = flare::experimental::erf(d_z(i));
            d_erfcx(i) = flare::experimental::erfcx(d_z(i));
        }

        FLARE_INLINE_FUNCTION
        void operator()(const TestRealErfcxTag &, const int & /*i*/) const {
            d_erfcx_dbl(0) = flare::experimental::erfcx(d_x(0));
            d_erfcx_dbl(1) = flare::experimental::erfcx(d_x(1));
            d_erfcx_dbl(2) = flare::experimental::erfcx(d_x(2));
            d_erfcx_dbl(3) = flare::experimental::erfcx(d_x(3));
            d_erfcx_dbl(4) = flare::experimental::erfcx(d_x(4));
            d_erfcx_dbl(5) = flare::experimental::erfcx(d_x(5));
        }
    };

    template<class ExecSpace>
    struct TestComplexBesselJ0Y0Function {
        using TensorType = flare::Tensor<flare::complex<double> *, ExecSpace>;
        using HostTensorType =
                flare::Tensor<flare::complex<double> *, flare::HostSpace>;

        TensorType d_z, d_cbj0, d_cby0;
        typename TensorType::HostMirror h_z, h_cbj0, h_cby0;
        HostTensorType h_ref_cbj0, h_ref_cby0;

        TensorType d_z_large, d_cbj0_large, d_cby0_large;
        typename TensorType::HostMirror h_z_large, h_cbj0_large, h_cby0_large;
        HostTensorType h_ref_cbj0_large, h_ref_cby0_large;

        void testit() {
            using flare::experimental::infinity;

            int N = 25;
            d_z = TensorType("d_z", N);
            d_cbj0 = TensorType("d_cbj0", N);
            d_cby0 = TensorType("d_cby0", N);
            h_z = flare::create_mirror_tensor(d_z);
            h_cbj0 = flare::create_mirror_tensor(d_cbj0);
            h_cby0 = flare::create_mirror_tensor(d_cby0);
            h_ref_cbj0 = HostTensorType("h_ref_cbj0", N);
            h_ref_cby0 = HostTensorType("h_ref_cby0", N);

            // Generate test inputs
            h_z(0) = flare::complex<double>(0.0, 0.0);
            // abs(z)<=25
            h_z(1) = flare::complex<double>(3.0, 2.0);
            h_z(2) = flare::complex<double>(3.0, -2.0);
            h_z(3) = flare::complex<double>(-3.0, 2.0);
            h_z(4) = flare::complex<double>(-3.0, -2.0);
            h_z(5) = flare::complex<double>(23.0, 10.0);
            h_z(6) = flare::complex<double>(23.0, -10.0);
            h_z(7) = flare::complex<double>(-23.0, 10.0);
            h_z(8) = flare::complex<double>(-23.0, -10.0);
            h_z(9) = flare::complex<double>(3.0, 0.0);
            h_z(10) = flare::complex<double>(-3.0, 0.0);
            h_z(11) = flare::complex<double>(23.0, 0.0);
            h_z(12) = flare::complex<double>(-23.0, 0.0);
            // abs(z)>25
            h_z(13) = flare::complex<double>(28.0, 10.0);
            h_z(14) = flare::complex<double>(28.0, -10.0);
            h_z(15) = flare::complex<double>(-28.0, 10.0);
            h_z(16) = flare::complex<double>(-28.0, -10.0);
            h_z(17) = flare::complex<double>(60.0, 10.0);
            h_z(18) = flare::complex<double>(60.0, -10.0);
            h_z(19) = flare::complex<double>(-60.0, 10.0);
            h_z(20) = flare::complex<double>(-60.0, -10.0);
            h_z(21) = flare::complex<double>(28.0, 0.0);
            h_z(22) = flare::complex<double>(-28.0, 0.0);
            h_z(23) = flare::complex<double>(60.0, 0.0);
            h_z(24) = flare::complex<double>(-60.0, 0.0);

            flare::deep_copy(d_z, h_z);

            // Call Bessel functions
            using Property = flare::experimental::WorkItemProperty::None_t;
            flare::parallel_for(flare::RangePolicy<ExecSpace, Property>(0, N), *this);
            flare::fence();

            flare::deep_copy(h_cbj0, d_cbj0);
            flare::deep_copy(h_cby0, d_cby0);

            // Reference values computed with Octave
            h_ref_cbj0(0) = flare::complex<double>(1.000000000000000e+00, 0);
            h_ref_cbj0(1) =
                    flare::complex<double>(-1.249234879607422e+00, -9.479837920577351e-01);
            h_ref_cbj0(2) =
                    flare::complex<double>(-1.249234879607422e+00, +9.479837920577351e-01);
            h_ref_cbj0(3) =
                    flare::complex<double>(-1.249234879607422e+00, +9.479837920577351e-01);
            h_ref_cbj0(4) =
                    flare::complex<double>(-1.249234879607422e+00, -9.479837920577351e-01);
            h_ref_cbj0(5) =
                    flare::complex<double>(-1.602439981218195e+03, +7.230667451989807e+02);
            h_ref_cbj0(6) =
                    flare::complex<double>(-1.602439981218195e+03, -7.230667451989807e+02);
            h_ref_cbj0(7) =
                    flare::complex<double>(-1.602439981218195e+03, -7.230667451989807e+02);
            h_ref_cbj0(8) =
                    flare::complex<double>(-1.602439981218195e+03, +7.230667451989807e+02);
            h_ref_cbj0(9) = flare::complex<double>(-2.600519549019335e-01, 0);
            h_ref_cbj0(10) =
                    flare::complex<double>(-2.600519549019335e-01, +9.951051106466461e-18);
            h_ref_cbj0(11) = flare::complex<double>(-1.624127813134866e-01, 0);
            h_ref_cbj0(12) =
                    flare::complex<double>(-1.624127813134866e-01, -1.387778780781446e-17);
            h_ref_cbj0(13) =
                    flare::complex<double>(-1.012912188513958e+03, -1.256239636146142e+03);
            h_ref_cbj0(14) =
                    flare::complex<double>(-1.012912188513958e+03, +1.256239636146142e+03);
            h_ref_cbj0(15) =
                    flare::complex<double>(-1.012912188513958e+03, +1.256239636146142e+03);
            h_ref_cbj0(16) =
                    flare::complex<double>(-1.012912188513958e+03, -1.256239636146142e+03);
            h_ref_cbj0(17) =
                    flare::complex<double>(-1.040215134669324e+03, -4.338202386810095e+02);
            h_ref_cbj0(18) =
                    flare::complex<double>(-1.040215134669324e+03, +4.338202386810095e+02);
            h_ref_cbj0(19) =
                    flare::complex<double>(-1.040215134669324e+03, +4.338202386810095e+02);
            h_ref_cbj0(20) =
                    flare::complex<double>(-1.040215134669324e+03, -4.338202386810095e+02);
            h_ref_cbj0(21) = flare::complex<double>(-7.315701054899962e-02, 0);
            h_ref_cbj0(22) =
                    flare::complex<double>(-7.315701054899962e-02, -6.938893903907228e-18);
            h_ref_cbj0(23) = flare::complex<double>(-9.147180408906189e-02, 0);
            h_ref_cbj0(24) =
                    flare::complex<double>(-9.147180408906189e-02, +1.387778780781446e-17);

            h_ref_cby0(0) = flare::complex<double>(-infinity<double>::value, 0);
            h_ref_cby0(1) =
                    flare::complex<double>(1.000803196554890e+00, -1.231441609303427e+00);
            h_ref_cby0(2) =
                    flare::complex<double>(1.000803196554890e+00, +1.231441609303427e+00);
            h_ref_cby0(3) =
                    flare::complex<double>(-8.951643875605797e-01, -1.267028149911417e+00);
            h_ref_cby0(4) =
                    flare::complex<double>(-8.951643875605797e-01, +1.267028149911417e+00);
            h_ref_cby0(5) =
                    flare::complex<double>(-7.230667452992603e+02, -1.602439974000479e+03);
            h_ref_cby0(6) =
                    flare::complex<double>(-7.230667452992603e+02, +1.602439974000479e+03);
            h_ref_cby0(7) =
                    flare::complex<double>(7.230667450987011e+02, -1.602439988435912e+03);
            h_ref_cby0(8) =
                    flare::complex<double>(7.230667450987011e+02, +1.602439988435912e+03);
            h_ref_cby0(9) = flare::complex<double>(3.768500100127903e-01, 0);
            h_ref_cby0(10) =
                    flare::complex<double>(3.768500100127903e-01, -5.201039098038670e-01);
            h_ref_cby0(11) = flare::complex<double>(-3.598179027370283e-02, 0);
            h_ref_cby0(12) =
                    flare::complex<double>(-3.598179027370282e-02, -3.248255626269732e-01);
            h_ref_cby0(13) =
                    flare::complex<double>(1.256239642409530e+03, -1.012912186329053e+03);
            h_ref_cby0(14) =
                    flare::complex<double>(1.256239642409530e+03, +1.012912186329053e+03);
            h_ref_cby0(15) =
                    flare::complex<double>(-1.256239629882755e+03, -1.012912190698863e+03);
            h_ref_cby0(16) =
                    flare::complex<double>(-1.256239629882755e+03, +1.012912190698863e+03);
            h_ref_cby0(17) =
                    flare::complex<double>(4.338202411482646e+02, -1.040215130736213e+03);
            h_ref_cby0(18) =
                    flare::complex<double>(4.338202411482646e+02, +1.040215130736213e+03);
            h_ref_cby0(19) =
                    flare::complex<double>(-4.338202362137545e+02, -1.040215138602435e+03);
            h_ref_cby0(20) =
                    flare::complex<double>(-4.338202362137545e+02, +1.040215138602435e+03);
            h_ref_cby0(21) = flare::complex<double>(1.318364704235323e-01, 0);
            h_ref_cby0(22) =
                    flare::complex<double>(1.318364704235323e-01, -1.463140210979992e-01);
            h_ref_cby0(23) = flare::complex<double>(4.735895220944939e-02, 0);
            h_ref_cby0(24) =
                    flare::complex<double>(4.735895220944938e-02, -1.829436081781237e-01);

            for (int i = 0; i < N; i++) {
                REQUIRE_LE(flare::abs(h_cbj0(i) - h_ref_cbj0(i)),
                           flare::abs(h_ref_cbj0(i)) * 1e-13);
            }

            REQUIRE_EQ(h_ref_cby0(0), h_cby0(0));
            for (int i = 1; i < N; i++) {
                REQUIRE_LE(flare::abs(h_cby0(i) - h_ref_cby0(i)),
                           flare::abs(h_ref_cby0(i)) * 1e-13);
            }

            ////Test large arguments
            d_z_large = TensorType("d_z_large", 6);
            d_cbj0_large = TensorType("d_cbj0_large", 6);
            d_cby0_large = TensorType("d_cby0_large", 6);
            h_z_large = flare::create_mirror_tensor(d_z_large);
            h_cbj0_large = flare::create_mirror_tensor(d_cbj0_large);
            h_cby0_large = flare::create_mirror_tensor(d_cby0_large);
            h_ref_cbj0_large = HostTensorType("h_ref_cbj0_large", 2);
            h_ref_cby0_large = HostTensorType("h_ref_cby0_large", 2);

            h_z_large(0) = flare::complex<double>(10000.0, 100.0);
            h_z_large(1) = flare::complex<double>(10000.0, 100.0);
            h_z_large(2) = flare::complex<double>(10000.0, 100.0);
            h_z_large(3) = flare::complex<double>(-10000.0, 100.0);
            h_z_large(4) = flare::complex<double>(-10000.0, 100.0);
            h_z_large(5) = flare::complex<double>(-10000.0, 100.0);

            flare::deep_copy(d_z_large, h_z_large);

            flare::parallel_for(
                    flare::RangePolicy<ExecSpace, Property, TestLargeArgTag>(0, 1), *this);
            flare::fence();

            flare::deep_copy(h_cbj0_large, d_cbj0_large);
            flare::deep_copy(h_cby0_large, d_cby0_large);

            h_ref_cbj0_large(0) =
                    flare::complex<double>(-9.561811498244175e+40, -4.854995782103029e+40);
            h_ref_cbj0_large(1) =
                    flare::complex<double>(-9.561811498244175e+40, +4.854995782103029e+40);

            h_ref_cby0_large(0) =
                    flare::complex<double>(4.854995782103029e+40, -9.561811498244175e+40);
            h_ref_cby0_large(1) =
                    flare::complex<double>(-4.854995782103029e+40, -9.561811498244175e+40);

            REQUIRE(((flare::abs(h_cbj0_large(0) - h_ref_cbj0_large(0)) <
                      flare::abs(h_ref_cbj0_large(0)) * 1e-12) &&
                     (flare::abs(h_cbj0_large(0) - h_ref_cbj0_large(0)) >
                      flare::abs(h_ref_cbj0_large(0)) * 1e-13)));
            REQUIRE(flare::abs(h_cbj0_large(1) - h_ref_cbj0_large(0)) >
                    flare::abs(h_ref_cbj0_large(0)) * 1e-6);
            REQUIRE(flare::abs(h_cbj0_large(2) - h_ref_cbj0_large(0)) <
                    flare::abs(h_ref_cbj0_large(0)) * 1e-13);
            REQUIRE(((flare::abs(h_cbj0_large(3) - h_ref_cbj0_large(1)) <
                      flare::abs(h_ref_cbj0_large(1)) * 1e-12) &&
                     (flare::abs(h_cbj0_large(3) - h_ref_cbj0_large(1)) >
                      flare::abs(h_ref_cbj0_large(1)) * 1e-13)));
            REQUIRE(flare::abs(h_cbj0_large(4) - h_ref_cbj0_large(1)) >
                    flare::abs(h_ref_cbj0_large(1)) * 1e-6);
            REQUIRE(flare::abs(h_cbj0_large(5) - h_ref_cbj0_large(1)) <
                    flare::abs(h_ref_cbj0_large(1)) * 1e-13);

            REQUIRE(((flare::abs(h_cby0_large(0) - h_ref_cby0_large(0)) <
                      flare::abs(h_ref_cby0_large(0)) * 1e-12) &&
                     (flare::abs(h_cby0_large(0) - h_ref_cby0_large(0)) >
                      flare::abs(h_ref_cby0_large(0)) * 1e-13)));
            REQUIRE(flare::abs(h_cby0_large(1) - h_ref_cby0_large(0)) >
                    flare::abs(h_ref_cby0_large(0)) * 1e-6);
            REQUIRE(flare::abs(h_cby0_large(2) - h_ref_cby0_large(0)) <
                    flare::abs(h_ref_cby0_large(0)) * 1e-13);
            REQUIRE(((flare::abs(h_cby0_large(3) - h_ref_cby0_large(1)) <
                      flare::abs(h_ref_cby0_large(1)) * 1e-12) &&
                     (flare::abs(h_cby0_large(3) - h_ref_cby0_large(1)) >
                      flare::abs(h_ref_cby0_large(1)) * 1e-13)));
            REQUIRE(flare::abs(h_cby0_large(4) - h_ref_cby0_large(1)) >
                    flare::abs(h_ref_cby0_large(1)) * 1e-6);
            REQUIRE(flare::abs(h_cby0_large(5) - h_ref_cby0_large(1)) <
                    flare::abs(h_ref_cby0_large(1)) * 1e-13);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const int &i) const {
            d_cbj0(i) = flare::experimental::cyl_bessel_j0<flare::complex<double>,
                    double, int>(d_z(i));
            d_cby0(i) = flare::experimental::cyl_bessel_y0<flare::complex<double>,
                    double, int>(d_z(i));
        }

        FLARE_INLINE_FUNCTION
        void operator()(const TestLargeArgTag &, const int & /*i*/) const {
            d_cbj0_large(0) =
                    flare::experimental::cyl_bessel_j0<flare::complex<double>, double,
                            int>(d_z_large(0));
            d_cbj0_large(1) =
                    flare::experimental::cyl_bessel_j0<flare::complex<double>, double,
                            int>(d_z_large(1), 11000, 3000);
            d_cbj0_large(2) =
                    flare::experimental::cyl_bessel_j0<flare::complex<double>, double,
                            int>(d_z_large(2), 11000, 7500);
            d_cbj0_large(3) =
                    flare::experimental::cyl_bessel_j0<flare::complex<double>, double,
                            int>(d_z_large(3));
            d_cbj0_large(4) =
                    flare::experimental::cyl_bessel_j0<flare::complex<double>, double,
                            int>(d_z_large(4), 11000, 3000);
            d_cbj0_large(5) =
                    flare::experimental::cyl_bessel_j0<flare::complex<double>, double,
                            int>(d_z_large(5), 11000, 7500);

            d_cby0_large(0) =
                    flare::experimental::cyl_bessel_y0<flare::complex<double>, double,
                            int>(d_z_large(0));
            d_cby0_large(1) =
                    flare::experimental::cyl_bessel_y0<flare::complex<double>, double,
                            int>(d_z_large(1), 11000, 3000);
            d_cby0_large(2) =
                    flare::experimental::cyl_bessel_y0<flare::complex<double>, double,
                            int>(d_z_large(2), 11000, 7500);
            d_cby0_large(3) =
                    flare::experimental::cyl_bessel_y0<flare::complex<double>, double,
                            int>(d_z_large(3));
            d_cby0_large(4) =
                    flare::experimental::cyl_bessel_y0<flare::complex<double>, double,
                            int>(d_z_large(4), 11000, 3000);
            d_cby0_large(5) =
                    flare::experimental::cyl_bessel_y0<flare::complex<double>, double,
                            int>(d_z_large(5), 11000, 7500);
        }
    };

    template<class ExecSpace>
    struct TestComplexBesselJ1Y1Function {
        using TensorType = flare::Tensor<flare::complex<double> *, ExecSpace>;
        using HostTensorType =
                flare::Tensor<flare::complex<double> *, flare::HostSpace>;

        TensorType d_z, d_cbj1, d_cby1;
        typename TensorType::HostMirror h_z, h_cbj1, h_cby1;
        HostTensorType h_ref_cbj1, h_ref_cby1;

        TensorType d_z_large, d_cbj1_large, d_cby1_large;
        typename TensorType::HostMirror h_z_large, h_cbj1_large, h_cby1_large;
        HostTensorType h_ref_cbj1_large, h_ref_cby1_large;

        void testit() {
            using flare::experimental::infinity;

            int N = 25;
            d_z = TensorType("d_z", N);
            d_cbj1 = TensorType("d_cbj1", N);
            d_cby1 = TensorType("d_cby1", N);
            h_z = flare::create_mirror_tensor(d_z);
            h_cbj1 = flare::create_mirror_tensor(d_cbj1);
            h_cby1 = flare::create_mirror_tensor(d_cby1);
            h_ref_cbj1 = HostTensorType("h_ref_cbj1", N);
            h_ref_cby1 = HostTensorType("h_ref_cby1", N);

            // Generate test inputs
            h_z(0) = flare::complex<double>(0.0, 0.0);
            // abs(z)<=25
            h_z(1) = flare::complex<double>(3.0, 2.0);
            h_z(2) = flare::complex<double>(3.0, -2.0);
            h_z(3) = flare::complex<double>(-3.0, 2.0);
            h_z(4) = flare::complex<double>(-3.0, -2.0);
            h_z(5) = flare::complex<double>(23.0, 10.0);
            h_z(6) = flare::complex<double>(23.0, -10.0);
            h_z(7) = flare::complex<double>(-23.0, 10.0);
            h_z(8) = flare::complex<double>(-23.0, -10.0);
            h_z(9) = flare::complex<double>(3.0, 0.0);
            h_z(10) = flare::complex<double>(-3.0, 0.0);
            h_z(11) = flare::complex<double>(23.0, 0.0);
            h_z(12) = flare::complex<double>(-23.0, 0.0);
            // abs(z)>25
            h_z(13) = flare::complex<double>(28.0, 10.0);
            h_z(14) = flare::complex<double>(28.0, -10.0);
            h_z(15) = flare::complex<double>(-28.0, 10.0);
            h_z(16) = flare::complex<double>(-28.0, -10.0);
            h_z(17) = flare::complex<double>(60.0, 10.0);
            h_z(18) = flare::complex<double>(60.0, -10.0);
            h_z(19) = flare::complex<double>(-60.0, 10.0);
            h_z(20) = flare::complex<double>(-60.0, -10.0);
            h_z(21) = flare::complex<double>(28.0, 0.0);
            h_z(22) = flare::complex<double>(-28.0, 0.0);
            h_z(23) = flare::complex<double>(60.0, 0.0);
            h_z(24) = flare::complex<double>(-60.0, 0.0);

            flare::deep_copy(d_z, h_z);

            // Call Bessel functions
            using Property = flare::experimental::WorkItemProperty::None_t;
            flare::parallel_for(flare::RangePolicy<ExecSpace, Property>(0, N), *this);
            flare::fence();

            flare::deep_copy(h_cbj1, d_cbj1);
            flare::deep_copy(h_cby1, d_cby1);

            // Reference values computed with Octave
            h_ref_cbj1(0) = flare::complex<double>(0, 0);
            h_ref_cbj1(1) =
                    flare::complex<double>(7.801488485792540e-01, -1.260982060238848e+00);
            h_ref_cbj1(2) =
                    flare::complex<double>(7.801488485792540e-01, +1.260982060238848e+00);
            h_ref_cbj1(3) =
                    flare::complex<double>(-7.801488485792543e-01, -1.260982060238848e+00);
            h_ref_cbj1(4) =
                    flare::complex<double>(-7.801488485792543e-01, +1.260982060238848e+00);
            h_ref_cbj1(5) =
                    flare::complex<double>(-7.469476253429664e+02, -1.576608505254311e+03);
            h_ref_cbj1(6) =
                    flare::complex<double>(-7.469476253429664e+02, +1.576608505254311e+03);
            h_ref_cbj1(7) =
                    flare::complex<double>(7.469476253429661e+02, -1.576608505254311e+03);
            h_ref_cbj1(8) =
                    flare::complex<double>(7.469476253429661e+02, +1.576608505254311e+03);
            h_ref_cbj1(9) = flare::complex<double>(3.390589585259365e-01, 0);
            h_ref_cbj1(10) =
                    flare::complex<double>(-3.390589585259365e-01, +3.373499138396203e-17);
            h_ref_cbj1(11) = flare::complex<double>(-3.951932188370151e-02, 0);
            h_ref_cbj1(12) =
                    flare::complex<double>(3.951932188370151e-02, +7.988560221984213e-18);
            h_ref_cbj1(13) =
                    flare::complex<double>(1.233147100257312e+03, -1.027302265904111e+03);
            h_ref_cbj1(14) =
                    flare::complex<double>(1.233147100257312e+03, +1.027302265904111e+03);
            h_ref_cbj1(15) =
                    flare::complex<double>(-1.233147100257312e+03, -1.027302265904111e+03);
            h_ref_cbj1(16) =
                    flare::complex<double>(-1.233147100257312e+03, +1.027302265904111e+03);
            h_ref_cbj1(17) =
                    flare::complex<double>(4.248029136732908e+02, -1.042364939115052e+03);
            h_ref_cbj1(18) =
                    flare::complex<double>(4.248029136732908e+02, +1.042364939115052e+03);
            h_ref_cbj1(19) =
                    flare::complex<double>(-4.248029136732909e+02, -1.042364939115052e+03);
            h_ref_cbj1(20) =
                    flare::complex<double>(-4.248029136732909e+02, +1.042364939115052e+03);
            h_ref_cbj1(21) = flare::complex<double>(1.305514883350938e-01, 0);
            h_ref_cbj1(22) =
                    flare::complex<double>(-1.305514883350938e-01, +7.993709105806192e-18);
            h_ref_cbj1(23) = flare::complex<double>(4.659838375816632e-02, 0);
            h_ref_cbj1(24) =
                    flare::complex<double>(-4.659838375816632e-02, +6.322680793358811e-18);

            h_ref_cby1(0) = flare::complex<double>(-infinity<double>::value, 0);
            h_ref_cby1(1) =
                    flare::complex<double>(1.285849341463599e+00, +7.250812532419394e-01);
            h_ref_cby1(2) =
                    flare::complex<double>(1.285849341463599e+00, -7.250812532419394e-01);
            h_ref_cby1(3) =
                    flare::complex<double>(1.236114779014097e+00, -8.352164439165690e-01);
            h_ref_cby1(4) =
                    flare::complex<double>(1.236114779014097e+00, +8.352164439165690e-01);
            h_ref_cby1(5) =
                    flare::complex<double>(1.576608512528508e+03, -7.469476251109801e+02);
            h_ref_cby1(6) =
                    flare::complex<double>(1.576608512528508e+03, +7.469476251109801e+02);
            h_ref_cby1(7) =
                    flare::complex<double>(1.576608497980113e+03, +7.469476255749524e+02);
            h_ref_cby1(8) =
                    flare::complex<double>(1.576608497980113e+03, -7.469476255749524e+02);
            h_ref_cby1(9) = flare::complex<double>(3.246744247918000e-01, 0);
            h_ref_cby1(10) =
                    flare::complex<double>(-3.246744247918000e-01, -6.781179170518730e-01);
            h_ref_cby1(11) = flare::complex<double>(1.616692009926331e-01, 0);
            h_ref_cby1(12) =
                    flare::complex<double>(-1.616692009926332e-01, +7.903864376740302e-02);
            h_ref_cby1(13) =
                    flare::complex<double>(1.027302268200224e+03, +1.233147093992241e+03);
            h_ref_cby1(14) =
                    flare::complex<double>(1.027302268200224e+03, -1.233147093992241e+03);
            h_ref_cby1(15) =
                    flare::complex<double>(1.027302263607999e+03, -1.233147106522383e+03);
            h_ref_cby1(16) =
                    flare::complex<double>(1.027302263607999e+03, +1.233147106522383e+03);
            h_ref_cby1(17) =
                    flare::complex<double>(1.042364943073579e+03, +4.248029112344685e+02);
            h_ref_cby1(18) =
                    flare::complex<double>(1.042364943073579e+03, -4.248029112344685e+02);
            h_ref_cby1(19) =
                    flare::complex<double>(1.042364935156525e+03, -4.248029161121132e+02);
            h_ref_cby1(20) =
                    flare::complex<double>(1.042364935156525e+03, +4.248029161121132e+02);
            h_ref_cby1(21) = flare::complex<double>(7.552212658226459e-02, 0);
            h_ref_cby1(22) =
                    flare::complex<double>(-7.552212658226459e-02, -2.611029766701876e-01);
            h_ref_cby1(23) = flare::complex<double>(9.186960936986688e-02, 0);
            h_ref_cby1(24) =
                    flare::complex<double>(-9.186960936986688e-02, -9.319676751633262e-02);

            for (int i = 0; i < N; i++) {
                REQUIRE_LE(flare::abs(h_cbj1(i) - h_ref_cbj1(i)),
                           flare::abs(h_ref_cbj1(i)) * 1e-13);
            }

            REQUIRE_EQ(h_ref_cby1(0), h_cby1(0));
            for (int i = 1; i < N; i++) {
                REQUIRE_LE(flare::abs(h_cby1(i) - h_ref_cby1(i)),
                           flare::abs(h_ref_cby1(i)) * 1e-13);
            }

            ////Test large arguments
            d_z_large = TensorType("d_z_large", 6);
            d_cbj1_large = TensorType("d_cbj1_large", 6);
            d_cby1_large = TensorType("d_cby1_large", 6);
            h_z_large = flare::create_mirror_tensor(d_z_large);
            h_cbj1_large = flare::create_mirror_tensor(d_cbj1_large);
            h_cby1_large = flare::create_mirror_tensor(d_cby1_large);
            h_ref_cbj1_large = HostTensorType("h_ref_cbj1_large", 2);
            h_ref_cby1_large = HostTensorType("h_ref_cby1_large", 2);

            h_z_large(0) = flare::complex<double>(10000.0, 100.0);
            h_z_large(1) = flare::complex<double>(10000.0, 100.0);
            h_z_large(2) = flare::complex<double>(10000.0, 100.0);
            h_z_large(3) = flare::complex<double>(-10000.0, 100.0);
            h_z_large(4) = flare::complex<double>(-10000.0, 100.0);
            h_z_large(5) = flare::complex<double>(-10000.0, 100.0);

            flare::deep_copy(d_z_large, h_z_large);

            flare::parallel_for(
                    flare::RangePolicy<ExecSpace, Property, TestLargeArgTag>(0, 1), *this);
            flare::fence();

            flare::deep_copy(h_cbj1_large, d_cbj1_large);
            flare::deep_copy(h_cby1_large, d_cby1_large);

            h_ref_cbj1_large(0) =
                    flare::complex<double>(4.854515317906369e+40, -9.562049455402486e+40);
            h_ref_cbj1_large(1) =
                    flare::complex<double>(-4.854515317906371e+40, -9.562049455402486e+40);

            h_ref_cby1_large(0) =
                    flare::complex<double>(9.562049455402486e+40, 4.854515317906369e+40);
            h_ref_cby1_large(1) =
                    flare::complex<double>(9.562049455402486e+40, -4.854515317906369e+40);

            REQUIRE(((flare::abs(h_cbj1_large(0) - h_ref_cbj1_large(0)) <
                      flare::abs(h_ref_cbj1_large(0)) * 1e-12) &&
                     (flare::abs(h_cbj1_large(0) - h_ref_cbj1_large(0)) >
                      flare::abs(h_ref_cbj1_large(0)) * 1e-13)));
            REQUIRE(flare::abs(h_cbj1_large(1) - h_ref_cbj1_large(0)) >
                    flare::abs(h_ref_cbj1_large(0)) * 1e-6);
            REQUIRE(flare::abs(h_cbj1_large(2) - h_ref_cbj1_large(0)) <
                    flare::abs(h_ref_cbj1_large(0)) * 1e-13);
            REQUIRE(((flare::abs(h_cbj1_large(3) - h_ref_cbj1_large(1)) <
                      flare::abs(h_ref_cbj1_large(1)) * 1e-12) &&
                     (flare::abs(h_cbj1_large(3) - h_ref_cbj1_large(1)) >
                      flare::abs(h_ref_cbj1_large(1)) * 1e-13)));
            REQUIRE(flare::abs(h_cbj1_large(4) - h_ref_cbj1_large(1)) >
                    flare::abs(h_ref_cbj1_large(1)) * 1e-6);
            REQUIRE(flare::abs(h_cbj1_large(5) - h_ref_cbj1_large(1)) <
                    flare::abs(h_ref_cbj1_large(1)) * 1e-13);

            REQUIRE(((flare::abs(h_cby1_large(0) - h_ref_cby1_large(0)) <
                      flare::abs(h_ref_cby1_large(0)) * 1e-12) &&
                     (flare::abs(h_cby1_large(0) - h_ref_cby1_large(0)) >
                      flare::abs(h_ref_cby1_large(0)) * 1e-13)));
            REQUIRE(flare::abs(h_cby1_large(1) - h_ref_cby1_large(0)) >
                    flare::abs(h_ref_cby1_large(0)) * 1e-6);
            REQUIRE(flare::abs(h_cby1_large(2) - h_ref_cby1_large(0)) <
                    flare::abs(h_ref_cby1_large(0)) * 1e-13);
            REQUIRE(((flare::abs(h_cby1_large(3) - h_ref_cby1_large(1)) <
                      flare::abs(h_ref_cby1_large(1)) * 1e-12) &&
                     (flare::abs(h_cby1_large(3) - h_ref_cby1_large(1)) >
                      flare::abs(h_ref_cby1_large(1)) * 1e-13)));
            REQUIRE(flare::abs(h_cby1_large(4) - h_ref_cby1_large(1)) >
                    flare::abs(h_ref_cby1_large(1)) * 1e-6);
            REQUIRE(flare::abs(h_cby1_large(5) - h_ref_cby1_large(1)) <
                    flare::abs(h_ref_cby1_large(1)) * 1e-13);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const int &i) const {
            d_cbj1(i) = flare::experimental::cyl_bessel_j1<flare::complex<double>,
                    double, int>(d_z(i));
            d_cby1(i) = flare::experimental::cyl_bessel_y1<flare::complex<double>,
                    double, int>(d_z(i));
        }

        FLARE_INLINE_FUNCTION
        void operator()(const TestLargeArgTag &, const int & /*i*/) const {
            d_cbj1_large(0) =
                    flare::experimental::cyl_bessel_j1<flare::complex<double>, double,
                            int>(d_z_large(0));
            d_cbj1_large(1) =
                    flare::experimental::cyl_bessel_j1<flare::complex<double>, double,
                            int>(d_z_large(1), 11000, 3000);
            d_cbj1_large(2) =
                    flare::experimental::cyl_bessel_j1<flare::complex<double>, double,
                            int>(d_z_large(2), 11000, 7500);
            d_cbj1_large(3) =
                    flare::experimental::cyl_bessel_j1<flare::complex<double>, double,
                            int>(d_z_large(3));
            d_cbj1_large(4) =
                    flare::experimental::cyl_bessel_j1<flare::complex<double>, double,
                            int>(d_z_large(4), 11000, 3000);
            d_cbj1_large(5) =
                    flare::experimental::cyl_bessel_j1<flare::complex<double>, double,
                            int>(d_z_large(5), 11000, 7500);

            d_cby1_large(0) =
                    flare::experimental::cyl_bessel_y1<flare::complex<double>, double,
                            int>(d_z_large(0));
            d_cby1_large(1) =
                    flare::experimental::cyl_bessel_y1<flare::complex<double>, double,
                            int>(d_z_large(1), 11000, 3000);
            d_cby1_large(2) =
                    flare::experimental::cyl_bessel_y1<flare::complex<double>, double,
                            int>(d_z_large(2), 11000, 7500);
            d_cby1_large(3) =
                    flare::experimental::cyl_bessel_y1<flare::complex<double>, double,
                            int>(d_z_large(3));
            d_cby1_large(4) =
                    flare::experimental::cyl_bessel_y1<flare::complex<double>, double,
                            int>(d_z_large(4), 11000, 3000);
            d_cby1_large(5) =
                    flare::experimental::cyl_bessel_y1<flare::complex<double>, double,
                            int>(d_z_large(5), 11000, 7500);
        }
    };

    template<class ExecSpace>
    struct TestComplexBesselI0K0Function {
        using TensorType = flare::Tensor<flare::complex<double> *, ExecSpace>;
        using HostTensorType =
                flare::Tensor<flare::complex<double> *, flare::HostSpace>;

        TensorType d_z, d_cbi0, d_cbk0;
        typename TensorType::HostMirror h_z, h_cbi0, h_cbk0;
        HostTensorType h_ref_cbi0, h_ref_cbk0;

        TensorType d_z_large, d_cbi0_large, d_cbk0_large;
        typename TensorType::HostMirror h_z_large, h_cbi0_large, h_cbk0_large;
        HostTensorType h_ref_cbi0_large, h_ref_cbk0_large;

        void testit() {
            using flare::experimental::infinity;

            int N = 25;
            d_z = TensorType("d_z", N);
            d_cbi0 = TensorType("d_cbi0", N);
            d_cbk0 = TensorType("d_cbk0", N);
            h_z = flare::create_mirror_tensor(d_z);
            h_cbi0 = flare::create_mirror_tensor(d_cbi0);
            h_cbk0 = flare::create_mirror_tensor(d_cbk0);
            h_ref_cbi0 = HostTensorType("h_ref_cbi0", N);
            h_ref_cbk0 = HostTensorType("h_ref_cbk0", N);

            // Generate test inputs
            h_z(0) = flare::complex<double>(0.0, 0.0);
            h_z(1) = flare::complex<double>(3.0, 2.0);
            h_z(2) = flare::complex<double>(3.0, -2.0);
            h_z(3) = flare::complex<double>(-3.0, 2.0);
            h_z(4) = flare::complex<double>(-3.0, -2.0);
            h_z(5) = flare::complex<double>(23.0, 10.0);
            h_z(6) = flare::complex<double>(23.0, -10.0);
            h_z(7) = flare::complex<double>(-23.0, 10.0);
            h_z(8) = flare::complex<double>(-23.0, -10.0);
            h_z(9) = flare::complex<double>(3.0, 0.0);
            h_z(10) = flare::complex<double>(-3.0, 0.0);
            h_z(11) = flare::complex<double>(23.0, 0.0);
            h_z(12) = flare::complex<double>(-23.0, 0.0);
            h_z(13) = flare::complex<double>(28.0, 10.0);
            h_z(14) = flare::complex<double>(28.0, -10.0);
            h_z(15) = flare::complex<double>(-28.0, 10.0);
            h_z(16) = flare::complex<double>(-28.0, -10.0);
            h_z(17) = flare::complex<double>(60.0, 10.0);
            h_z(18) = flare::complex<double>(60.0, -10.0);
            h_z(19) = flare::complex<double>(-60.0, 10.0);
            h_z(20) = flare::complex<double>(-60.0, -10.0);
            h_z(21) = flare::complex<double>(28.0, 0.0);
            h_z(22) = flare::complex<double>(-28.0, 0.0);
            h_z(23) = flare::complex<double>(60.0, 0.0);
            h_z(24) = flare::complex<double>(-60.0, 0.0);

            flare::deep_copy(d_z, h_z);

            // Call Bessel functions
            using Property = flare::experimental::WorkItemProperty::None_t;
            flare::parallel_for(flare::RangePolicy<ExecSpace, Property>(0, N), *this);
            flare::fence();

            flare::deep_copy(h_cbi0, d_cbi0);
            flare::deep_copy(h_cbk0, d_cbk0);

            // Reference values computed with Octave
            h_ref_cbi0(0) = flare::complex<double>(1.000000000000000e+00, 0);
            h_ref_cbi0(1) =
                    flare::complex<double>(-4.695171920440706e-01, +4.313788409468920e+00);
            h_ref_cbi0(2) =
                    flare::complex<double>(-4.695171920440706e-01, -4.313788409468920e+00);
            h_ref_cbi0(3) =
                    flare::complex<double>(-4.695171920440706e-01, -4.313788409468920e+00);
            h_ref_cbi0(4) =
                    flare::complex<double>(-4.695171920440706e-01, +4.313788409468920e+00);
            h_ref_cbi0(5) =
                    flare::complex<double>(-7.276526052028507e+08, -2.806354803468570e+08);
            h_ref_cbi0(6) =
                    flare::complex<double>(-7.276526052028507e+08, +2.806354803468570e+08);
            h_ref_cbi0(7) =
                    flare::complex<double>(-7.276526052028507e+08, +2.806354803468570e+08);
            h_ref_cbi0(8) =
                    flare::complex<double>(-7.276526052028507e+08, -2.806354803468570e+08);
            h_ref_cbi0(9) = flare::complex<double>(4.880792585865025e+00, 0);
            h_ref_cbi0(10) = flare::complex<double>(4.880792585865025e+00, 0);
            h_ref_cbi0(11) = flare::complex<double>(8.151421225128924e+08, 0);
            h_ref_cbi0(12) = flare::complex<double>(8.151421225128924e+08, 0);
            h_ref_cbi0(13) =
                    flare::complex<double>(-9.775983282455373e+10, -4.159160389327644e+10);
            h_ref_cbi0(14) =
                    flare::complex<double>(-9.775983282455373e+10, +4.159160389327644e+10);
            h_ref_cbi0(15) =
                    flare::complex<double>(-9.775983282455373e+10, +4.159160389327644e+10);
            h_ref_cbi0(16) =
                    flare::complex<double>(-9.775983282455373e+10, -4.159160389327644e+10);
            h_ref_cbi0(17) =
                    flare::complex<double>(-5.158377566681892e+24, -2.766704059464302e+24);
            h_ref_cbi0(18) =
                    flare::complex<double>(-5.158377566681892e+24, +2.766704059464302e+24);
            h_ref_cbi0(19) =
                    flare::complex<double>(-5.158377566681892e+24, +2.766704059464302e+24);
            h_ref_cbi0(20) =
                    flare::complex<double>(-5.158377566681892e+24, -2.766704059464302e+24);
            h_ref_cbi0(21) = flare::complex<double>(1.095346047317573e+11, 0);
            h_ref_cbi0(22) = flare::complex<double>(1.095346047317573e+11, 0);
            h_ref_cbi0(23) = flare::complex<double>(5.894077055609803e+24, 0);
            h_ref_cbi0(24) = flare::complex<double>(5.894077055609803e+24, 0);

            h_ref_cbk0(0) = flare::complex<double>(infinity<double>::value, 0);
            h_ref_cbk0(1) =
                    flare::complex<double>(-2.078722558742977e-02, -2.431266356716766e-02);
            h_ref_cbk0(2) =
                    flare::complex<double>(-2.078722558742977e-02, +2.431266356716766e-02);
            h_ref_cbk0(3) =
                    flare::complex<double>(-1.357295320191579e+01, +1.499344424826928e+00);
            h_ref_cbk0(4) =
                    flare::complex<double>(-1.357295320191579e+01, -1.499344424826928e+00);
            h_ref_cbk0(5) =
                    flare::complex<double>(-1.820476218131465e-11, +1.795056004780177e-11);
            h_ref_cbk0(6) =
                    flare::complex<double>(-1.820476218131465e-11, -1.795056004780177e-11);
            h_ref_cbk0(7) =
                    flare::complex<double>(8.816423633943287e+08, +2.285988078870750e+09);
            h_ref_cbk0(8) =
                    flare::complex<double>(8.816423633943287e+08, -2.285988078870750e+09);
            h_ref_cbk0(9) = flare::complex<double>(3.473950438627926e-02, 0);
            h_ref_cbk0(10) =
                    flare::complex<double>(3.473950438627926e-02, -1.533346213144909e+01);
            h_ref_cbk0(11) = flare::complex<double>(2.667545110351910e-11, 0);
            h_ref_cbk0(12) =
                    flare::complex<double>(2.667545110351910e-11, -2.560844503718094e+09);
            h_ref_cbk0(13) =
                    flare::complex<double>(-1.163319528590747e-13, +1.073711234918388e-13);
            h_ref_cbk0(14) =
                    flare::complex<double>(-1.163319528590747e-13, -1.073711234918388e-13);
            h_ref_cbk0(15) =
                    flare::complex<double>(1.306638772421339e+11, +3.071215726177843e+11);
            h_ref_cbk0(16) =
                    flare::complex<double>(1.306638772421339e+11, -3.071215726177843e+11);
            h_ref_cbk0(17) =
                    flare::complex<double>(-1.111584549467388e-27, +8.581979311477652e-28);
            h_ref_cbk0(18) =
                    flare::complex<double>(-1.111584549467388e-27, -8.581979311477652e-28);
            h_ref_cbk0(19) =
                    flare::complex<double>(8.691857147870108e+24, +1.620552106793022e+25);
            h_ref_cbk0(20) =
                    flare::complex<double>(8.691857147870108e+24, -1.620552106793022e+25);
            h_ref_cbk0(21) = flare::complex<double>(1.630534586888181e-13, 0);
            h_ref_cbk0(22) =
                    flare::complex<double>(1.630534586888181e-13, -3.441131095391506e+11);
            h_ref_cbk0(23) = flare::complex<double>(1.413897840559108e-27, 0);
            h_ref_cbk0(24) =
                    flare::complex<double>(1.413897840559108e-27, -1.851678917759592e+25);

            for (int i = 0; i < N; i++) {
                REQUIRE_LE(flare::abs(h_cbi0(i) - h_ref_cbi0(i)),
                           flare::abs(h_ref_cbi0(i)) * 1e-13);
            }

            REQUIRE_EQ(h_ref_cbk0(0), h_cbk0(0));
            for (int i = 1; i < N; i++) {
                REQUIRE_LE(flare::abs(h_cbk0(i) - h_ref_cbk0(i)),
                           flare::abs(h_ref_cbk0(i)) * 1e-13);
            }

            ////Test large arguments
            d_z_large = TensorType("d_z_large", 6);
            d_cbi0_large = TensorType("d_cbi0_large", 6);
            h_z_large = flare::create_mirror_tensor(d_z_large);
            h_cbi0_large = flare::create_mirror_tensor(d_cbi0_large);
            h_ref_cbi0_large = HostTensorType("h_ref_cbi0_large", 2);

            h_z_large(0) = flare::complex<double>(100.0, 10.0);
            h_z_large(1) = flare::complex<double>(100.0, 10.0);
            h_z_large(2) = flare::complex<double>(100.0, 10.0);
            h_z_large(3) = flare::complex<double>(-100.0, 10.0);
            h_z_large(4) = flare::complex<double>(-100.0, 10.0);
            h_z_large(5) = flare::complex<double>(-100.0, 10.0);

            flare::deep_copy(d_z_large, h_z_large);

            flare::parallel_for(
                    flare::RangePolicy<ExecSpace, Property, TestLargeArgTag>(0, 1), *this);
            flare::fence();

            flare::deep_copy(h_cbi0_large, d_cbi0_large);

            h_ref_cbi0_large(0) =
                    flare::complex<double>(-9.266819049505678e+41, -5.370779383266049e+41);
            h_ref_cbi0_large(1) =
                    flare::complex<double>(-9.266819049505678e+41, +5.370779383266049e+41);

            REQUIRE(flare::abs(h_cbi0_large(0) - h_ref_cbi0_large(0)) <
                    flare::abs(h_ref_cbi0_large(0)) * 1e-15);
            REQUIRE(flare::abs(h_cbi0_large(1) - h_ref_cbi0_large(0)) >
                    flare::abs(h_ref_cbi0_large(0)) * 1e-4);
            REQUIRE(flare::abs(h_cbi0_large(2) - h_ref_cbi0_large(0)) <
                    flare::abs(h_ref_cbi0_large(0)) * 1e-15);
            REQUIRE(flare::abs(h_cbi0_large(3) - h_ref_cbi0_large(1)) <
                    flare::abs(h_ref_cbi0_large(1)) * 1e-15);
            REQUIRE(flare::abs(h_cbi0_large(4) - h_ref_cbi0_large(1)) >
                    flare::abs(h_ref_cbi0_large(1)) * 1e-4);
            REQUIRE(flare::abs(h_cbi0_large(5) - h_ref_cbi0_large(1)) <
                    flare::abs(h_ref_cbi0_large(1)) * 1e-15);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const int &i) const {
            d_cbi0(i) = flare::experimental::cyl_bessel_i0<flare::complex<double>,
                    double, int>(d_z(i));
            d_cbk0(i) = flare::experimental::cyl_bessel_k0<flare::complex<double>,
                    double, int>(d_z(i));
        }

        FLARE_INLINE_FUNCTION
        void operator()(const TestLargeArgTag &, const int & /*i*/) const {
            d_cbi0_large(0) =
                    flare::experimental::cyl_bessel_i0<flare::complex<double>, double,
                            int>(d_z_large(0));
            d_cbi0_large(1) =
                    flare::experimental::cyl_bessel_i0<flare::complex<double>, double,
                            int>(d_z_large(1), 110, 35);
            d_cbi0_large(2) =
                    flare::experimental::cyl_bessel_i0<flare::complex<double>, double,
                            int>(d_z_large(2), 110, 190);
            d_cbi0_large(3) =
                    flare::experimental::cyl_bessel_i0<flare::complex<double>, double,
                            int>(d_z_large(3));
            d_cbi0_large(4) =
                    flare::experimental::cyl_bessel_i0<flare::complex<double>, double,
                            int>(d_z_large(4), 110, 35);
            d_cbi0_large(5) =
                    flare::experimental::cyl_bessel_i0<flare::complex<double>, double,
                            int>(d_z_large(5), 110, 190);
        }
    };

    template<class ExecSpace>
    struct TestComplexBesselI1K1Function {
        using TensorType = flare::Tensor<flare::complex<double> *, ExecSpace>;
        using HostTensorType =
                flare::Tensor<flare::complex<double> *, flare::HostSpace>;

        TensorType d_z, d_cbi1, d_cbk1;
        typename TensorType::HostMirror h_z, h_cbi1, h_cbk1;
        HostTensorType h_ref_cbi1, h_ref_cbk1;

        TensorType d_z_large, d_cbi1_large, d_cbk1_large;
        typename TensorType::HostMirror h_z_large, h_cbi1_large, h_cbk1_large;
        HostTensorType h_ref_cbi1_large, h_ref_cbk1_large;

        void testit() {
            using flare::experimental::infinity;

            int N = 25;
            d_z = TensorType("d_z", N);
            d_cbi1 = TensorType("d_cbi1", N);
            d_cbk1 = TensorType("d_cbk1", N);
            h_z = flare::create_mirror_tensor(d_z);
            h_cbi1 = flare::create_mirror_tensor(d_cbi1);
            h_cbk1 = flare::create_mirror_tensor(d_cbk1);
            h_ref_cbi1 = HostTensorType("h_ref_cbi1", N);
            h_ref_cbk1 = HostTensorType("h_ref_cbk1", N);

            // Generate test inputs
            h_z(0) = flare::complex<double>(0.0, 0.0);
            h_z(1) = flare::complex<double>(3.0, 2.0);
            h_z(2) = flare::complex<double>(3.0, -2.0);
            h_z(3) = flare::complex<double>(-3.0, 2.0);
            h_z(4) = flare::complex<double>(-3.0, -2.0);
            h_z(5) = flare::complex<double>(23.0, 10.0);
            h_z(6) = flare::complex<double>(23.0, -10.0);
            h_z(7) = flare::complex<double>(-23.0, 10.0);
            h_z(8) = flare::complex<double>(-23.0, -10.0);
            h_z(9) = flare::complex<double>(3.0, 0.0);
            h_z(10) = flare::complex<double>(-3.0, 0.0);
            h_z(11) = flare::complex<double>(23.0, 0.0);
            h_z(12) = flare::complex<double>(-23.0, 0.0);
            h_z(13) = flare::complex<double>(28.0, 10.0);
            h_z(14) = flare::complex<double>(28.0, -10.0);
            h_z(15) = flare::complex<double>(-28.0, 10.0);
            h_z(16) = flare::complex<double>(-28.0, -10.0);
            h_z(17) = flare::complex<double>(60.0, 10.0);
            h_z(18) = flare::complex<double>(60.0, -10.0);
            h_z(19) = flare::complex<double>(-60.0, 10.0);
            h_z(20) = flare::complex<double>(-60.0, -10.0);
            h_z(21) = flare::complex<double>(28.0, 0.0);
            h_z(22) = flare::complex<double>(-28.0, 0.0);
            h_z(23) = flare::complex<double>(60.0, 0.0);
            h_z(24) = flare::complex<double>(-60.0, 0.0);

            flare::deep_copy(d_z, h_z);

            // Call Bessel functions
            using Property = flare::experimental::WorkItemProperty::None_t;
            flare::parallel_for(flare::RangePolicy<ExecSpace, Property>(0, N), *this);
            flare::fence();

            flare::deep_copy(h_cbi1, d_cbi1);
            flare::deep_copy(h_cbk1, d_cbk1);

            // Reference values computed with Octave
            h_ref_cbi1(0) = flare::complex<double>(0, 0);
            h_ref_cbi1(1) =
                    flare::complex<double>(-8.127809410735776e-01, +3.780682961371298e+00);
            h_ref_cbi1(2) =
                    flare::complex<double>(-8.127809410735776e-01, -3.780682961371298e+00);
            h_ref_cbi1(3) =
                    flare::complex<double>(8.127809410735776e-01, +3.780682961371298e+00);
            h_ref_cbi1(4) =
                    flare::complex<double>(8.127809410735776e-01, -3.780682961371298e+00);
            h_ref_cbi1(5) =
                    flare::complex<double>(-7.119745937677552e+08, -2.813616375214342e+08);
            h_ref_cbi1(6) =
                    flare::complex<double>(-7.119745937677552e+08, +2.813616375214342e+08);
            h_ref_cbi1(7) =
                    flare::complex<double>(7.119745937677552e+08, -2.813616375214342e+08);
            h_ref_cbi1(8) =
                    flare::complex<double>(7.119745937677552e+08, +2.813616375214342e+08);
            h_ref_cbi1(9) = flare::complex<double>(3.953370217402609e+00, 0);
            h_ref_cbi1(10) = flare::complex<double>(-3.953370217402609e+00, 0);
            h_ref_cbi1(11) = flare::complex<double>(7.972200260896506e+08, 0);
            h_ref_cbi1(12) = flare::complex<double>(-7.972200260896506e+08, 0);
            h_ref_cbi1(13) =
                    flare::complex<double>(-9.596150723281404e+10, -4.149038020045121e+10);
            h_ref_cbi1(14) =
                    flare::complex<double>(-9.596150723281404e+10, +4.149038020045121e+10);
            h_ref_cbi1(15) =
                    flare::complex<double>(9.596150723281404e+10, -4.149038020045121e+10);
            h_ref_cbi1(16) =
                    flare::complex<double>(9.596150723281404e+10, +4.149038020045121e+10);
            h_ref_cbi1(17) =
                    flare::complex<double>(-5.112615594220387e+24, -2.751210232069100e+24);
            h_ref_cbi1(18) =
                    flare::complex<double>(-5.112615594220387e+24, +2.751210232069100e+24);
            h_ref_cbi1(19) =
                    flare::complex<double>(5.112615594220387e+24, -2.751210232069100e+24);
            h_ref_cbi1(20) =
                    flare::complex<double>(5.112615594220387e+24, +2.751210232069100e+24);
            h_ref_cbi1(21) = flare::complex<double>(1.075605042080823e+11, 0);
            h_ref_cbi1(22) = flare::complex<double>(-1.075605042080823e+11, 0);
            h_ref_cbi1(23) = flare::complex<double>(5.844751588390470e+24, 0);
            h_ref_cbi1(24) = flare::complex<double>(-5.844751588390470e+24, 0);

            h_ref_cbk1(0) = flare::complex<double>(infinity<double>::value, 0);
            h_ref_cbk1(1) =
                    flare::complex<double>(-2.480952007015153e-02, -2.557074905635180e-02);
            h_ref_cbk1(2) =
                    flare::complex<double>(-2.480952007015153e-02, +2.557074905635180e-02);
            h_ref_cbk1(3) =
                    flare::complex<double>(-1.185255629692602e+01, +2.527855884398198e+00);
            h_ref_cbk1(4) =
                    flare::complex<double>(-1.185255629692602e+01, -2.527855884398198e+00);
            h_ref_cbk1(5) =
                    flare::complex<double>(-1.839497240093994e-11, +1.841855854336314e-11);
            h_ref_cbk1(6) =
                    flare::complex<double>(-1.839497240093994e-11, -1.841855854336314e-11);
            h_ref_cbk1(7) =
                    flare::complex<double>(8.839236534393319e+08, +2.236734153323357e+09);
            h_ref_cbk1(8) =
                    flare::complex<double>(8.839236534393319e+08, -2.236734153323357e+09);
            h_ref_cbk1(9) = flare::complex<double>(4.015643112819419e-02, 0);
            h_ref_cbk1(10) =
                    flare::complex<double>(-4.015643112819419e-02, -1.241987883191272e+01);
            h_ref_cbk1(11) = flare::complex<double>(2.724930589574976e-11, 0);
            h_ref_cbk1(12) =
                    flare::complex<double>(-2.724930589574976e-11, -2.504540577257910e+09);
            h_ref_cbk1(13) =
                    flare::complex<double>(-1.175637676331817e-13, +1.097080943197297e-13);
            h_ref_cbk1(14) =
                    flare::complex<double>(-1.175637676331817e-13, -1.097080943197297e-13);
            h_ref_cbk1(15) =
                    flare::complex<double>(1.303458736323849e+11, +3.014719661500124e+11);
            h_ref_cbk1(16) =
                    flare::complex<double>(1.303458736323849e+11, -3.014719661500124e+11);
            h_ref_cbk1(17) =
                    flare::complex<double>(-1.119411861396158e-27, +8.666195226392352e-28);
            h_ref_cbk1(18) =
                    flare::complex<double>(-1.119411861396158e-27, -8.666195226392352e-28);
            h_ref_cbk1(19) =
                    flare::complex<double>(8.643181853549355e+24, +1.606175559143138e+25);
            h_ref_cbk1(20) =
                    flare::complex<double>(8.643181853549355e+24, -1.606175559143138e+25);
            h_ref_cbk1(21) = flare::complex<double>(1.659400107332009e-13, 0);
            h_ref_cbk1(22) =
                    flare::complex<double>(-1.659400107332009e-13, -3.379112898365253e+11);
            h_ref_cbk1(23) = flare::complex<double>(1.425632026517104e-27, 0);
            h_ref_cbk1(24) =
                    flare::complex<double>(-1.425632026517104e-27, -1.836182865214478e+25);

            for (int i = 0; i < N; i++) {
                REQUIRE_LE(flare::abs(h_cbi1(i) - h_ref_cbi1(i)),
                           flare::abs(h_ref_cbi1(i)) * 1e-13);
            }

            REQUIRE_EQ(h_ref_cbk1(0), h_cbk1(0));
            for (int i = 1; i < N; i++) {
                REQUIRE_LE(flare::abs(h_cbk1(i) - h_ref_cbk1(i)),
                           flare::abs(h_ref_cbk1(i)) * 1e-13);
            }

            ////Test large arguments
            d_z_large = TensorType("d_z_large", 6);
            d_cbi1_large = TensorType("d_cbi1_large", 6);
            h_z_large = flare::create_mirror_tensor(d_z_large);
            h_cbi1_large = flare::create_mirror_tensor(d_cbi1_large);
            h_ref_cbi1_large = HostTensorType("h_ref_cbi1_large", 2);

            h_z_large(0) = flare::complex<double>(100.0, 10.0);
            h_z_large(1) = flare::complex<double>(100.0, 10.0);
            h_z_large(2) = flare::complex<double>(100.0, 10.0);
            h_z_large(3) = flare::complex<double>(-100.0, 10.0);
            h_z_large(4) = flare::complex<double>(-100.0, 10.0);
            h_z_large(5) = flare::complex<double>(-100.0, 10.0);

            flare::deep_copy(d_z_large, h_z_large);

            flare::parallel_for(
                    flare::RangePolicy<ExecSpace, Property, TestLargeArgTag>(0, 1), *this);
            flare::fence();

            flare::deep_copy(h_cbi1_large, d_cbi1_large);

            h_ref_cbi1_large(0) =
                    flare::complex<double>(-9.218158020154234e+41, -5.348736158968607e+41);
            h_ref_cbi1_large(1) =
                    flare::complex<double>(9.218158020154234e+41, -5.348736158968607e+41);

            REQUIRE(flare::abs(h_cbi1_large(0) - h_ref_cbi1_large(0)) <
                    flare::abs(h_ref_cbi1_large(0)) * 1e-15);
            REQUIRE(flare::abs(h_cbi1_large(1) - h_ref_cbi1_large(0)) >
                    flare::abs(h_ref_cbi1_large(0)) * 1e-4);
            REQUIRE(flare::abs(h_cbi1_large(2) - h_ref_cbi1_large(0)) <
                    flare::abs(h_ref_cbi1_large(0)) * 1e-15);
            REQUIRE(flare::abs(h_cbi1_large(3) - h_ref_cbi1_large(1)) <
                    flare::abs(h_ref_cbi1_large(1)) * 1e-15);
            REQUIRE(flare::abs(h_cbi1_large(4) - h_ref_cbi1_large(1)) >
                    flare::abs(h_ref_cbi1_large(1)) * 1e-4);
            REQUIRE(flare::abs(h_cbi1_large(5) - h_ref_cbi1_large(1)) <
                    flare::abs(h_ref_cbi1_large(1)) * 1e-15);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const int &i) const {
            d_cbi1(i) = flare::experimental::cyl_bessel_i1<flare::complex<double>,
                    double, int>(d_z(i));
            d_cbk1(i) = flare::experimental::cyl_bessel_k1<flare::complex<double>,
                    double, int>(d_z(i));
        }

        FLARE_INLINE_FUNCTION
        void operator()(const TestLargeArgTag &, const int & /*i*/) const {
            d_cbi1_large(0) =
                    flare::experimental::cyl_bessel_i1<flare::complex<double>, double,
                            int>(d_z_large(0));
            d_cbi1_large(1) =
                    flare::experimental::cyl_bessel_i1<flare::complex<double>, double,
                            int>(d_z_large(1), 110, 35);
            d_cbi1_large(2) =
                    flare::experimental::cyl_bessel_i1<flare::complex<double>, double,
                            int>(d_z_large(2), 110, 190);
            d_cbi1_large(3) =
                    flare::experimental::cyl_bessel_i1<flare::complex<double>, double,
                            int>(d_z_large(3));
            d_cbi1_large(4) =
                    flare::experimental::cyl_bessel_i1<flare::complex<double>, double,
                            int>(d_z_large(4), 110, 35);
            d_cbi1_large(5) =
                    flare::experimental::cyl_bessel_i1<flare::complex<double>, double,
                            int>(d_z_large(5), 110, 190);
        }
    };

    template<class ExecSpace>
    struct TestComplexBesselH1Function {
        using TensorType = flare::Tensor<flare::complex<double> *, ExecSpace>;
        using HostTensorType =
                flare::Tensor<flare::complex<double> *, flare::HostSpace>;

        TensorType d_z, d_ch10, d_ch11;
        typename TensorType::HostMirror h_z, h_ch10, h_ch11;
        HostTensorType h_ref_ch10, h_ref_ch11;

        void testit() {
            using flare::experimental::infinity;

            int N = 25;
            d_z = TensorType("d_z", N);
            d_ch10 = TensorType("d_ch10", N);
            d_ch11 = TensorType("d_ch11", N);
            h_z = flare::create_mirror_tensor(d_z);
            h_ch10 = flare::create_mirror_tensor(d_ch10);
            h_ch11 = flare::create_mirror_tensor(d_ch11);
            h_ref_ch10 = HostTensorType("h_ref_ch10", N);
            h_ref_ch11 = HostTensorType("h_ref_ch11", N);

            // Generate test inputs
            h_z(0) = flare::complex<double>(0.0, 0.0);
            h_z(1) = flare::complex<double>(3.0, 2.0);
            h_z(2) = flare::complex<double>(3.0, -2.0);
            h_z(3) = flare::complex<double>(-3.0, 2.0);
            h_z(4) = flare::complex<double>(-3.0, -2.0);
            h_z(5) = flare::complex<double>(23.0, 10.0);
            h_z(6) = flare::complex<double>(23.0, -10.0);
            h_z(7) = flare::complex<double>(-23.0, 10.0);
            h_z(8) = flare::complex<double>(-23.0, -10.0);
            h_z(9) = flare::complex<double>(3.0, 0.0);
            h_z(10) = flare::complex<double>(-3.0, 0.0);
            h_z(11) = flare::complex<double>(23.0, 0.0);
            h_z(12) = flare::complex<double>(-23.0, 0.0);
            h_z(13) = flare::complex<double>(28.0, 10.0);
            h_z(14) = flare::complex<double>(28.0, -10.0);
            h_z(15) = flare::complex<double>(-28.0, 10.0);
            h_z(16) = flare::complex<double>(-28.0, -10.0);
            h_z(17) = flare::complex<double>(200.0, 60.0);
            h_z(18) = flare::complex<double>(200.0, -60.0);
            h_z(19) = flare::complex<double>(-200.0, 60.0);
            h_z(20) = flare::complex<double>(-200.0, -60.0);
            h_z(21) = flare::complex<double>(28.0, 0.0);
            h_z(22) = flare::complex<double>(-28.0, 0.0);
            h_z(23) = flare::complex<double>(200.0, 0.0);
            h_z(24) = flare::complex<double>(-200.0, 0.0);

            flare::deep_copy(d_z, h_z);

            // Call Hankel functions
            using Property = flare::experimental::WorkItemProperty::None_t;
            flare::parallel_for(flare::RangePolicy<ExecSpace, Property>(0, N), *this);
            flare::fence();

            flare::deep_copy(h_ch10, d_ch10);
            flare::deep_copy(h_ch11, d_ch11);

            // Reference values computed with Octave
            h_ref_ch10(0) = flare::complex<double>(1.0, -infinity<double>::value);
            h_ref_ch10(1) =
                    flare::complex<double>(-1.779327030399459e-02, +5.281940449715537e-02);
            h_ref_ch10(2) =
                    flare::complex<double>(-2.480676488910849e+00, +1.948786988612626e+00);
            h_ref_ch10(3) =
                    flare::complex<double>(1.779327030399459e-02, +5.281940449715537e-02);
            h_ref_ch10(4) =
                    flare::complex<double>(-2.516263029518839e+00, -1.843148179618315e+00);
            h_ref_ch10(5) =
                    flare::complex<double>(-7.217716938222564e-06, -1.002796203581228e-07);
            h_ref_ch10(6) =
                    flare::complex<double>(-3.204879955218674e+03, -1.446133490498241e+03);
            h_ref_ch10(7) =
                    flare::complex<double>(7.217716938222564e-06, -1.002796203581228e-07);
            h_ref_ch10(8) =
                    flare::complex<double>(-3.204879969654108e+03, +1.446133490297682e+03);
            h_ref_ch10(9) =
                    flare::complex<double>(-2.600519549019334e-01, +3.768500100127903e-01);
            h_ref_ch10(10) =
                    flare::complex<double>(2.600519549019334e-01, +3.768500100127903e-01);
            h_ref_ch10(11) =
                    flare::complex<double>(-1.624127813134865e-01, -3.598179027370283e-02);
            h_ref_ch10(12) =
                    flare::complex<double>(1.624127813134865e-01, -3.598179027370283e-02);
            h_ref_ch10(13) =
                    flare::complex<double>(-2.184905481759440e-06, +6.263387166445335e-06);
            h_ref_ch10(14) =
                    flare::complex<double>(-2.025824374843011e+03, +2.512479278555672e+03);
            h_ref_ch10(15) =
                    flare::complex<double>(2.184905481759440e-06, +6.263387166445335e-06);
            h_ref_ch10(16) =
                    flare::complex<double>(-2.025824379212821e+03, -2.512479266028897e+03);
            h_ref_ch10(17) =
                    flare::complex<double>(-1.983689762743337e-28, -4.408449940359881e-28);
            h_ref_ch10(18) =
                    flare::complex<double>(-8.261945332108929e+23, -6.252486138159269e+24);
            h_ref_ch10(19) =
                    flare::complex<double>(1.983689762743337e-28, -4.408449940359881e-28);
            h_ref_ch10(20) =
                    flare::complex<double>(-8.261945332108929e+23, +6.252486138159269e+24);
            h_ref_ch10(21) =
                    flare::complex<double>(-7.315701054899959e-02, +1.318364704235323e-01);
            h_ref_ch10(22) =
                    flare::complex<double>(7.315701054899959e-02, +1.318364704235323e-01);
            h_ref_ch10(23) =
                    flare::complex<double>(-1.543743993056510e-02, -5.426577524981793e-02);
            h_ref_ch10(24) =
                    flare::complex<double>(1.543743993056510e-02, -5.426577524981793e-02);

            h_ref_ch11(0) = flare::complex<double>(0.0, -infinity<double>::value);
            h_ref_ch11(1) =
                    flare::complex<double>(5.506759533731469e-02, +2.486728122475093e-02);
            h_ref_ch11(2) =
                    flare::complex<double>(1.505230101821194e+00, +2.546831401702448e+00);
            h_ref_ch11(3) =
                    flare::complex<double>(5.506759533731469e-02, -2.486728122475093e-02);
            h_ref_ch11(4) =
                    flare::complex<double>(-1.615365292495823e+00, +2.497096839252946e+00);
            h_ref_ch11(5) =
                    flare::complex<double>(-2.319863729607219e-07, +7.274197719836158e-06);
            h_ref_ch11(6) =
                    flare::complex<double>(-1.493895250453947e+03, +3.153217017782819e+03);
            h_ref_ch11(7) =
                    flare::complex<double>(-2.319863729607210e-07, -7.274197719836158e-06);
            h_ref_ch11(8) =
                    flare::complex<double>(1.493895250917918e+03, +3.153217003234423e+03);
            h_ref_ch11(9) =
                    flare::complex<double>(3.390589585259364e-01, +3.246744247918000e-01);
            h_ref_ch11(10) =
                    flare::complex<double>(3.390589585259364e-01, -3.246744247918000e-01);
            h_ref_ch11(11) =
                    flare::complex<double>(-3.951932188370152e-02, +1.616692009926331e-01);
            h_ref_ch11(12) =
                    flare::complex<double>(-3.951932188370151e-02, -1.616692009926331e-01);
            h_ref_ch11(13) =
                    flare::complex<double>(6.265071091331731e-06, +2.296112637347948e-06);
            h_ref_ch11(14) =
                    flare::complex<double>(2.466294194249553e+03, +2.054604534104335e+03);
            h_ref_ch11(15) =
                    flare::complex<double>(6.265071091331731e-06, -2.296112637347947e-06);
            h_ref_ch11(16) =
                    flare::complex<double>(-2.466294206779695e+03, +2.054604529512110e+03);
            h_ref_ch11(17) =
                    flare::complex<double>(-4.416040381930448e-28, +1.974955285825768e-28);
            h_ref_ch11(18) =
                    flare::complex<double>(-6.250095237987940e+24, +8.112776606830997e+23);
            h_ref_ch11(19) =
                    flare::complex<double>(-4.416040381930448e-28, -1.974955285825769e-28);
            h_ref_ch11(20) =
                    flare::complex<double>(6.250095237987940e+24, +8.112776606831005e+23);
            h_ref_ch11(21) =
                    flare::complex<double>(1.305514883350938e-01, +7.552212658226459e-02);
            h_ref_ch11(22) =
                    flare::complex<double>(1.305514883350938e-01, -7.552212658226456e-02);
            h_ref_ch11(23) =
                    flare::complex<double>(-5.430453818237824e-02, +1.530182458038999e-02);
            h_ref_ch11(24) =
                    flare::complex<double>(-5.430453818237824e-02, -1.530182458039000e-02);

            REQUIRE_EQ(h_ref_ch10(0), h_ch10(0));
            for (int i = 1; i < N; i++) {
                REQUIRE_LE(flare::abs(h_ch10(i) - h_ref_ch10(i)),
                           flare::abs(h_ref_ch10(i)) * 1e-13);
            }

            REQUIRE_EQ(h_ref_ch11(0), h_ch11(0));
            for (int i = 1; i < N; i++) {
                REQUIRE_LE(flare::abs(h_ch11(i) - h_ref_ch11(i)),
                           flare::abs(h_ref_ch11(i)) * 1e-13);
            }
        }

        FLARE_INLINE_FUNCTION
        void operator()(const int &i) const {
            d_ch10(i) = flare::experimental::cyl_bessel_h10(d_z(i));
            d_ch11(i) = flare::experimental::cyl_bessel_h11(d_z(i));
        }
    };

    template<class ExecSpace>
    struct TestComplexBesselH2Function {
        using TensorType = flare::Tensor<flare::complex<double> *, ExecSpace>;
        using HostTensorType =
                flare::Tensor<flare::complex<double> *, flare::HostSpace>;

        TensorType d_z, d_ch20, d_ch21;
        typename TensorType::HostMirror h_z, h_ch20, h_ch21;
        HostTensorType h_ref_ch20, h_ref_ch21;

        void testit() {
            using flare::experimental::infinity;

            int N = 25;
            d_z = TensorType("d_z", N);
            d_ch20 = TensorType("d_ch20", N);
            d_ch21 = TensorType("d_ch21", N);
            h_z = flare::create_mirror_tensor(d_z);
            h_ch20 = flare::create_mirror_tensor(d_ch20);
            h_ch21 = flare::create_mirror_tensor(d_ch21);
            h_ref_ch20 = HostTensorType("h_ref_ch20", N);
            h_ref_ch21 = HostTensorType("h_ref_ch21", N);

            // Generate test inputs
            h_z(0) = flare::complex<double>(0.0, 0.0);
            h_z(1) = flare::complex<double>(3.0, 2.0);
            h_z(2) = flare::complex<double>(3.0, -2.0);
            h_z(3) = flare::complex<double>(-3.0, 2.0);
            h_z(4) = flare::complex<double>(-3.0, -2.0);
            h_z(5) = flare::complex<double>(23.0, 10.0);
            h_z(6) = flare::complex<double>(23.0, -10.0);
            h_z(7) = flare::complex<double>(-23.0, 10.0);
            h_z(8) = flare::complex<double>(-23.0, -10.0);
            h_z(9) = flare::complex<double>(3.0, 0.0);
            h_z(10) = flare::complex<double>(-3.0, 0.0);
            h_z(11) = flare::complex<double>(23.0, 0.0);
            h_z(12) = flare::complex<double>(-23.0, 0.0);
            h_z(13) = flare::complex<double>(28.0, 10.0);
            h_z(14) = flare::complex<double>(28.0, -10.0);
            h_z(15) = flare::complex<double>(-28.0, 10.0);
            h_z(16) = flare::complex<double>(-28.0, -10.0);
            h_z(17) = flare::complex<double>(200.0, 60.0);
            h_z(18) = flare::complex<double>(200.0, -60.0);
            h_z(19) = flare::complex<double>(-200.0, 60.0);
            h_z(20) = flare::complex<double>(-200.0, -60.0);
            h_z(21) = flare::complex<double>(28.0, 0.0);
            h_z(22) = flare::complex<double>(-28.0, 0.0);
            h_z(23) = flare::complex<double>(200.0, 0.0);
            h_z(24) = flare::complex<double>(-200.0, 0.0);

            flare::deep_copy(d_z, h_z);

            // Call Hankel functions
            flare::parallel_for(flare::RangePolicy<ExecSpace>(0, N), *this);
            flare::fence();

            flare::deep_copy(h_ch20, d_ch20);
            flare::deep_copy(h_ch21, d_ch21);

            // Reference values computed with Octave
            h_ref_ch20(0) = flare::complex<double>(1.0, infinity<double>::value);
            h_ref_ch20(1) =
                    flare::complex<double>(-2.480676488910849e+00, -1.948786988612626e+00);
            h_ref_ch20(2) =
                    flare::complex<double>(-1.779327030399459e-02, -5.281940449715537e-02);
            h_ref_ch20(3) =
                    flare::complex<double>(-2.516263029518839e+00, +1.843148179618315e+00);
            h_ref_ch20(4) =
                    flare::complex<double>(1.779327030399459e-02, -5.281940449715537e-02);
            h_ref_ch20(5) =
                    flare::complex<double>(-3.204879955218674e+03, +1.446133490498241e+03);
            h_ref_ch20(6) =
                    flare::complex<double>(-7.217716938222564e-06, +1.002796203581228e-07);
            h_ref_ch20(7) =
                    flare::complex<double>(-3.204879969654108e+03, -1.446133490297682e+03);
            h_ref_ch20(8) =
                    flare::complex<double>(7.217716938222564e-06, +1.002796203581228e-07);
            h_ref_ch20(9) =
                    flare::complex<double>(-2.600519549019334e-01, -3.768500100127903e-01);
            h_ref_ch20(10) =
                    flare::complex<double>(-7.801558647058006e-01, -3.768500100127903e-01);
            h_ref_ch20(11) =
                    flare::complex<double>(-1.624127813134865e-01, +3.598179027370283e-02);
            h_ref_ch20(12) =
                    flare::complex<double>(-4.872383439404597e-01, +3.598179027370281e-02);
            h_ref_ch20(13) =
                    flare::complex<double>(-2.025824374843011e+03, -2.512479278555672e+03);
            h_ref_ch20(14) =
                    flare::complex<double>(-2.184905481759440e-06, -6.263387166445335e-06);
            h_ref_ch20(15) =
                    flare::complex<double>(-2.025824379212821e+03, +2.512479266028897e+03);
            h_ref_ch20(16) =
                    flare::complex<double>(2.184905481759440e-06, -6.263387166445335e-06);
            h_ref_ch20(17) =
                    flare::complex<double>(-8.261945332108929e+23, +6.252486138159269e+24);
            h_ref_ch20(18) =
                    flare::complex<double>(-1.983689762743337e-28, +4.408449940359881e-28);
            h_ref_ch20(19) =
                    flare::complex<double>(-8.261945332108929e+23, -6.252486138159269e+24);
            h_ref_ch20(20) =
                    flare::complex<double>(1.983689762743337e-28, +4.408449940359881e-28);
            h_ref_ch20(21) =
                    flare::complex<double>(-7.315701054899959e-02, -1.318364704235323e-01);
            h_ref_ch20(22) =
                    flare::complex<double>(-2.194710316469988e-01, -1.318364704235323e-01);
            h_ref_ch20(23) =
                    flare::complex<double>(-1.543743993056510e-02, +5.426577524981793e-02);
            h_ref_ch20(24) =
                    flare::complex<double>(-4.631231979169528e-02, +5.426577524981793e-02);

            h_ref_ch21(0) = flare::complex<double>(0.0, infinity<double>::value);
            h_ref_ch21(1) =
                    flare::complex<double>(1.505230101821194e+00, -2.546831401702448e+00);
            h_ref_ch21(2) =
                    flare::complex<double>(5.506759533731469e-02, -2.486728122475093e-02);
            h_ref_ch21(3) =
                    flare::complex<double>(-1.615365292495823e+00, -2.497096839252946e+00);
            h_ref_ch21(4) =
                    flare::complex<double>(5.506759533731469e-02, +2.486728122475093e-02);
            h_ref_ch21(5) =
                    flare::complex<double>(-1.493895250453947e+03, -3.153217017782819e+03);
            h_ref_ch21(6) =
                    flare::complex<double>(-2.319863729607219e-07, -7.274197719836158e-06);
            h_ref_ch21(7) =
                    flare::complex<double>(1.493895250917918e+03, -3.153217003234423e+03);
            h_ref_ch21(8) =
                    flare::complex<double>(-2.319863729607210e-07, +7.274197719836158e-06);
            h_ref_ch21(9) =
                    flare::complex<double>(3.390589585259364e-01, -3.246744247918000e-01);
            h_ref_ch21(10) =
                    flare::complex<double>(-1.017176875577809e+00, +3.246744247918000e-01);
            h_ref_ch21(11) =
                    flare::complex<double>(-3.951932188370152e-02, -1.616692009926331e-01);
            h_ref_ch21(12) =
                    flare::complex<double>(1.185579656511045e-01, +1.616692009926332e-01);
            h_ref_ch21(13) =
                    flare::complex<double>(2.466294194249553e+03, -2.054604534104335e+03);
            h_ref_ch21(14) =
                    flare::complex<double>(6.265071091331731e-06, -2.296112637347948e-06);
            h_ref_ch21(15) =
                    flare::complex<double>(-2.466294206779695e+03, -2.054604529512110e+03);
            h_ref_ch21(16) =
                    flare::complex<double>(6.265071091331731e-06, +2.296112637347947e-06);
            h_ref_ch21(17) =
                    flare::complex<double>(-6.250095237987940e+24, -8.112776606830997e+23);
            h_ref_ch21(18) =
                    flare::complex<double>(-4.416040381930448e-28, -1.974955285825768e-28);
            h_ref_ch21(19) =
                    flare::complex<double>(6.250095237987940e+24, -8.112776606831005e+23);
            h_ref_ch21(20) =
                    flare::complex<double>(-4.416040381930448e-28, +1.974955285825769e-28);
            h_ref_ch21(21) =
                    flare::complex<double>(1.305514883350938e-01, -7.552212658226459e-02);
            h_ref_ch21(22) =
                    flare::complex<double>(-3.916544650052814e-01, +7.552212658226461e-02);
            h_ref_ch21(23) =
                    flare::complex<double>(-5.430453818237824e-02, -1.530182458038999e-02);
            h_ref_ch21(24) =
                    flare::complex<double>(1.629136145471347e-01, +1.530182458039000e-02);

            REQUIRE_EQ(h_ref_ch20(0), h_ch20(0));
            for (int i = 1; i < N; i++) {
                REQUIRE_LE(flare::abs(h_ch20(i) - h_ref_ch20(i)),
                           flare::abs(h_ref_ch20(i)) * 1e-13);
            }

            REQUIRE_EQ(h_ref_ch21(0), h_ch21(0));
            for (int i = 1; i < N; i++) {
                REQUIRE_LE(flare::abs(h_ch21(i) - h_ref_ch21(i)),
                           flare::abs(h_ref_ch21(i)) * 1e-13);
            }
        }

        FLARE_INLINE_FUNCTION
        void operator()(const int &i) const {
            d_ch20(i) = flare::experimental::cyl_bessel_h20(d_z(i));
            d_ch21(i) = flare::experimental::cyl_bessel_h21(d_z(i));
        }
    };

    TEST_CASE("TEST_CATEGORY, mathspecialfunc_expint1") {
        TestExponentialIntergral1Function<TEST_EXECSPACE> test;
        test.testit();
    }

    TEST_CASE("TEST_CATEGORY, mathspecialfunc_errorfunc") {
        TestComplexErrorFunction<TEST_EXECSPACE> test;
        test.testit();
    }

    TEST_CASE("TEST_CATEGORY, mathspecialfunc_cbesselj0y0") {
        TestComplexBesselJ0Y0Function<TEST_EXECSPACE> test;
        test.testit();
    }

    TEST_CASE("TEST_CATEGORY, mathspecialfunc_cbesselj1y1") {
        TestComplexBesselJ1Y1Function<TEST_EXECSPACE> test;
        test.testit();
    }

    TEST_CASE("TEST_CATEGORY, mathspecialfunc_cbesseli0k0") {
        TestComplexBesselI0K0Function<TEST_EXECSPACE> test;
        test.testit();
    }

    TEST_CASE("TEST_CATEGORY, mathspecialfunc_cbesseli1k1") {
        TestComplexBesselI1K1Function<TEST_EXECSPACE> test;
        test.testit();
    }

    TEST_CASE("TEST_CATEGORY, mathspecialfunc_cbesselh1stkind") {
        TestComplexBesselH1Function<TEST_EXECSPACE> test;
        test.testit();
    }

    TEST_CASE("TEST_CATEGORY, mathspecialfunc_cbesselh2ndkind") {
        TestComplexBesselH2Function<TEST_EXECSPACE> test;
        test.testit();
    }

}  // namespace Test
