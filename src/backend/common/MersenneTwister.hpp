// Copyright 2023 The EA Authors.
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

/********************************************************
 * Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University.
 * Copyright (c) 2011, 2012 Mutsuo Saito, Makoto Matsumoto, Hiroshima
 * University and University of Tokyo.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of the Hiroshima University, The Uinversity
 *       of Tokyo nor the names of its contributors may be used to
 *       endorse or promote products derived from this software without
 *       specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************/

/*
 * These numbers have been obtained from the following file :
 * https://github.com/MersenneTwister-Lab/MTGP/blob/master/cuda-sample/mtgp32dc-param-11213.c
 */

#pragma once

#include <fly/defines.h>

namespace flare {
namespace common {
const dim_t MaxBlocks     = 32;
const dim_t TableLength   = 16 * MaxBlocks;
const dim_t MersenneN     = 351;
const dim_t MtStateLength = MaxBlocks * MersenneN;

static unsigned pos[] = {
    88, 84, 25, 42, 22, 11, 76, 11, 42, 60, 45, 80, 81, 16, 63, 38,
    3,  55, 9,  75, 70, 63, 32, 70, 58, 33, 18, 9,  14, 91, 90, 86,
};

static unsigned sh1[] = {
    19, 15, 4, 20, 1,  16, 16, 15, 6,  6, 12, 6,  8,  1, 14, 28,
    30, 1,  9, 17, 15, 15, 7,  12, 21, 7, 7,  12, 16, 4, 10, 6,
};

static unsigned sh2[] = {
    5, 12, 18, 9, 5, 1,  6,  16, 11, 11, 13, 9,  18, 19, 18, 1,
    2, 16, 15, 6, 6, 17, 15, 10, 2,  10, 13, 13, 3,  2,  14, 7,
};

static unsigned mask = 4294443008;
// static const unsigned mask[] = {
// 4294443008, 4294443008, 4294443008, 4294443008, 4294443008, 4294443008,
// 4294443008, 4294443008, 4294443008, 4294443008, 4294443008, 4294443008,
// 4294443008, 4294443008, 4294443008, 4294443008, 4294443008, 4294443008,
// 4294443008, 4294443008, 4294443008, 4294443008, 4294443008, 4294443008,
// 4294443008, 4294443008, 4294443008, 4294443008, 4294443008, 4294443008,
// 4294443008, 4294443008,
//};

static unsigned recursion_tbl[] = {
    0,          2879706668, 3137826695, 279165355,  570425344,  2309281324,
    2567401351, 849590699,  38330,      2879669142, 3137862205, 279129105,
    570463674,  2309243798, 2567436861, 849554449,  0,          2593609479,
    1975655185, 4015344662, 357564441,  2412205854, 1620187912, 4194651151,
    53865,      2593621358, 1975699832, 4015365759, 357618288,  2412217719,
    1620232545, 4194672230, 0,          1350351572, 3879866427, 3074337519,
    1908408359, 566016755,  2525106204, 3338578632, 56684,      1350330296,
    3879914839, 3074324355, 1908464971, 565995423,  2525154672, 3338565540,
    0,          1055977248, 2682116586, 2704094922, 271581238,  784396054,
    2414729692, 2971481852, 23789,      1055962061, 2682094855, 2704108071,
    271604955,  784380923,  2414708017, 2971494929, 0,          2953117369,
    62376084,   3014866477, 122683469,  3075800820, 82299097,   3034789472,
    18916,      2953099101, 62357872,   3014885321, 122702249,  3075782416,
    82280765,   3034808196, 0,          1684682268, 3815655459, 2265218623,
    723517535,  1330263619, 3360573564, 2888072800, 51884,      1684733104,
    3815670415, 2265232531, 723569395,  1330314479, 3360588496, 2888086732,
    0,          465136444,  2086871972, 1742358680, 574619745,  972647261,
    1579361221, 1167739129, 19553,      465119069,  2086891461, 1742341369,
    574639104,  972629820,  1579380644, 1167721624, 0,          2761653723,
    3161842783, 418290052,  1969225846, 3522919853, 3373655081, 1838062066,
    50425,      2761668898, 3161792678, 418274685,  1969276047, 3522935124,
    3373605072, 1838046475, 0,          823868564,  2715863359, 2432431531,
    1012924546, 226180118,  2642463165, 2895901993, 46626,      823888566,
    2715844381, 2432385929, 1012971168, 226200116,  2642444191, 2895856395,
    0,          3163477696, 1302313789, 4044451325, 2389704861, 855561821,
    3287268256, 2137091424, 40970,      3163453130, 1302272823, 4044475895,
    2389745815, 855537239,  3287227306, 2137116010, 0,          112997176,
    3723016697, 3679750849, 4213178533, 4254872477, 650688860,  544508516,
    42220,      113022932,  3722976533, 3679727149, 4213220425, 4254898033,
    650649008,  544485000,  0,          612320333,  3070909104, 2473926397,
    3292528821, 3762242808, 1934252549, 1463098952, 21839,      612307202,
    3070889983, 2473937842, 3292550650, 3762229687, 1934233418, 1463110407,
    0,          3452614784, 3739161126, 320187046,  3513778374, 481998918,
    263131872,  3261442656, 46663,      3452571335, 3739198561, 320150753,
    3513824897, 481955329,  263169191,  3261406247, 0,          588293362,
    2445856652, 3000527742, 2311061713, 2865800227, 403230557,  991456175,
    62105,      588273259,  2445819157, 3000539623, 2311123528, 2865780410,
    403193284,  991467830,  0,          2177339548, 3971246860, 1836317584,
    4080009455, 1928826995, 528772067,  2655255423, 59650,      2177333662,
    3971252750, 1836257938, 4080069101, 1928821105, 528777953,  2655195773,
    0,          3849091899, 1973261595, 2431774240, 442499322,  4279008193,
    1878889953, 2324819674, 17524,      3849076559, 1973277039, 2431756884,
    442516622,  4278992821, 1878905237, 2324802222, 0,          2208474059,
    1390454348, 3510764935, 2555379979, 468886208,  3400574791, 1225917580,
    43685,      2208434542, 1390415081, 3510808354, 2555423662, 468846693,
    3400535522, 1225961001, 0,          753114312,  1662184071, 1341224527,
    2666529043, 2987630043, 4259507092, 3506534236, 32220,      753131796,
    1662162779, 1341197203, 2666560719, 2987646983, 4259485256, 3506506368,
    0,          578983141,  3246792315, 3808725662, 1649410346, 1087542735,
    2748718929, 2169801652, 22528,      578997477,  3246802555, 3808744094,
    1649432874, 1087557071, 2748729169, 2169820084, 0,          2174674396,
    1341825145, 3462677925, 3861905719, 1739515115, 2848629070, 676611218,
    35444,      2174644136, 1341794829, 3462713297, 3861941059, 1739484831,
    2848598842, 676646630,  0,          3312287941, 1270375145, 2396381740,
    2464153923, 1468891526, 3646448554, 473293679,  38325,      3312260464,
    1270413148, 2396354457, 2464191734, 1468863539, 3646486047, 473265882,
    0,          1011438676, 4087471600, 3488123300, 455082329,  661214477,
    3900824745, 3569912061, 52539,      1011456367, 4087419083, 3488105631,
    455134306,  661231670,  3900772754, 3569894854, 0,          3529350863,
    2972224887, 1668617144, 3278897504, 288202671,  1918405655, 2684687064,
    37845,      3529313562, 2972196514, 1668644973, 3278934709, 288164986,
    1918377922, 2684715277, 0,          1851876274, 2748239338, 3450842712,
    3060793725, 3625018063, 364825751,  2078256933, 54612,      1851897574,
    2748192958, 3450829580, 3060847657, 3625039771, 364779971,  2078243441,
    0,          2798128189, 889378840,  2479543333, 3664773513, 2092436916,
    4017281425, 1236981164, 52108,      2798176177, 889328532,  2479497129,
    3664824837, 2092484152, 4017230365, 1236934176, 0,          1067241239,
    3453461153, 4065029558, 4190110101, 3327970946, 873964340,  193686563,
    27698,      1067229989, 3453472403, 4065001860, 4190137767, 3327959728,
    873975558,  193658897,  0,          2213869621, 887682042,  3072068559,
    4061135267, 1910831510, 3338203737, 1158417004, 40265,      2213832060,
    887647923,  3072104070, 4061175018, 1910793439, 3338170128, 1158453029,
    0,          3310321543, 1761913168, 2890651351, 2243953084, 1083145787,
    3972311276, 697030507,  31607,      3310290160, 1761923623, 2890640800,
    2243984075, 1083114828, 3972322203, 697019420,  0,          2168967706,
    1652210191, 3812452373, 663749057,  2799162331, 1173011406, 3299699156,
    62673,      2168923851, 1652182750, 3812465860, 663811344,  2799118090,
    1172983583, 3299712261, 0,          2787116654, 1233955067, 4021071509,
    3287286227, 1708132285, 2323425576, 744271686,  37500,      2787152914,
    1233926791, 4021042409, 3287323567, 1708168641, 2323397460, 744242490,
    0,          3709953686, 1938447361, 2930457239, 1075839468, 2634114938,
    866803181,  4002102139, 50633,      3709969247, 1938463176, 2930507614,
    1075889189, 2634130099, 866818084,  4002152114, 0,          1104625460,
    154195956,  1223157952, 1706033657, 610746061,  1820382733, 760736057,
    35538,      1104655846, 154164518,  1223123474, 1706068779, 610776095,
    1820351711, 760701931,
};

static unsigned temper_tbl[] = {
    0,          101711872,  634912768,  600309760,  673972224,  775684096,
    234094592,  199491584,  855825920,  890428928,  383442432,  281730560,
    456056320,  490659328,  1056366080, 954654208,  0,          6581248,
    135327744,  141859840,  922910720,  929491968,  1058172928, 1064705024,
    5266944,    3420672,    138456576,  136626688,  928177664,  926331392,
    1061301760, 1059471872, 0,          69272064,   134479872,  203751936,
    289406976,  358679040,  423886848,  493158912,  544153088,  609098752,
    678108672,  743054336,  825171456,  890117120,  959127040,  1024072704,
    0,          608635904,  2523648,    610373120,  5439488,    605293568,
    7700992,    607292928,  274488832,  874205696,  276487168,  876466176,
    269442560,  877154816,  271178752,  879677440,  0,          1214875648,
    67174400,   1281918976, 1614020608, 677218304,  1681195008, 744261632,
    537288192,  1752159744, 604462592,  1819203072, 1077042688, 140236288,
    1144217088, 207279616,  0,          3567010304, 194572288,  3741626880,
    604008960,  4036767744, 798523904,  4211392512, 1563500032, 2309839872,
    1453977088, 2184555520, 2033282048, 2913807872, 1923718144, 2788548096,
    0,          136677376,  272809984,  409421824,  2109440,    134592512,
    274919424,  407336960,  1891638784, 2028312064, 1619189248, 1755796992,
    1893740032, 2026219008, 1621290496, 1753703936, 0,          4026617856,
    172764160,  4199382016, 75673600,   4102283264, 248421376,  4275031040,
    1158700544, 3037793792, 1331458560, 3210551808, 1100148224, 2979249664,
    1272889856, 3151991296, 0,          2147483648, 0,          2147483648,
    3221225472, 1073741824, 3221225472, 1073741824, 536878592,  2684362240,
    536878592,  2684362240, 3758104064, 1610620416, 3758104064, 1610620416,
    0,          3225420800, 2228224,    3227649024, 809369600,  4034790400,
    807141376,  4032562176, 1073880576, 2151815680, 1075846656, 2153781760,
    1882988032, 2960923136, 1881021952, 2958957056, 0,          206130176,
    1107561472, 1313685504, 537462784,  742409216,  1645020160, 1849968640,
    272670208,  470405632,  1380225536, 1577967104, 810128896,  1006688768,
    1917688320, 2114246144, 0,          807406080,  275513344,  541854208,
    1573888,    808979968,  276038656,  542379520,  269098496,  539628544,
    6692352,    809899008,  269621760,  540151808,  8264192,    811470848,
    0,          1612447744, 142082048,  1751384064, 7602176,    1617428480,
    135004160,  1745879040, 10493440,   1622941184, 148381184,  1757683200,
    13901312,   1623727616, 145497600,  1756372480, 0,          1275199488,
    2793406464, 3934388224, 352321536,  1493303296, 3011510272, 4286709760,
    689970688,  1696734720, 2409635328, 3282181632, 1008737792, 1881284096,
    2594184704, 3600948736, 0,          150999040,  33619968,   184619008,
    2103296,    153094144,  35723264,   186714112,  805543424,  956534272,
    839032320,  990023168,  807634432,  958633472,  841123328,  992122368,
    0,          3225487872, 3876324352, 659361280,  4198400000, 981436928,
    489849856,  3715337728, 545267200,  3770749952, 3347847680, 130879488,
    3669925376, 452957184,  1035115008, 4260597760, 0,          3910139904,
    2496659456, 2109734912, 3687579648, 853278720,  1327235072, 2785804288,
    164634112,  3770686976, 2634030592, 1947213312, 3525058048, 990649856,
    1187782144, 2950438400, 0,          201461760,  3812622336, 4014084096,
    33554432,   235016192,  3779067904, 3980529664, 1352670720, 1554124288,
    3017809408, 3219262976, 1386225152, 1587678720, 2984254976, 3185708544,
    0,          1469123584, 3497837568, 2280507392, 2818795520, 4287783936,
    2021633024, 804167680,  5381632,    1472401920, 3492731392, 2277496320,
    2823910912, 4290804224, 2016260608, 800898560,  0,          2550235136,
    537106432,  3087144960, 1074806784, 3625041920, 1611913216, 4161951744,
    212480,     2550316544, 536913408,  3087083008, 1075019264, 3625123328,
    1611720192, 4161889792, 0,          543432704,  623958016,  89454592,
    27283968,   566522368,  613452288,  83143168,   5381632,    540425728,
    627230208,  84338176,   32656384,   563506176,  616731648,  78033920,
    0,          134377472,  2521945088, 2656281600, 4236288,    138597376,
    2517725184, 2652045312, 7249408,    141356544,  2520730112, 2654812672,
    3029504,    137120256,  2524966400, 2659032576, 0,          269697536,
    42663936,   311968256,  1346895872, 1079722496, 1388511232, 1120944640,
    284057088,  16587776,   308633088,  41294848,   1084644864, 1354046464,
    1110269440, 1379802112, 0,          54056960,   5527552,    57442304,
    22155264,   40552448,   17188864,   37654528,   2153984,    51906048,
    7636480,    55336448,   24301056,   38409728,   19305984,   35540480,
    0,          3597599744, 21800448,   3609436672, 4467200,    3593154048,
    17337344,   3613886464, 209935872,  3672922624, 231733248,  3684760576,
    214397952,  3668471808, 227267072,  3689207296, 0,          70910976,
    7421952,    72041472,   272663552,  343572480,  271696896,  336314368,
    1879350784, 1950259712, 1886772736, 1951390208, 1615075840, 1685986816,
    1614109184, 1678728704, 0,          1761935360, 185210880,  1645156352,
    1101029376, 681926656,  1252685824, 598702080,  74128896,   1835933184,
    258016768,  1717831168, 1170963968, 751730176,  1321297408, 667182592,
    0,          3338677248, 1516357632, 2640438272, 302069760,  3573617664,
    1214312448, 2405489664, 3368033792, 264253952,  2460079616, 1436678656,
    3670091264, 499190272,  2158030336, 1201717760, 0,          554461696,
    7452672,    561893888,  3422689792, 3977146368, 3430130176, 3984574464,
    1102241280, 1623110656, 1103324672, 1624181760, 2377171968, 2898046464,
    2378267648, 2899121664, 0,          697340416,  7652864,    702829568,
    1074688000, 1772020224, 1081783808, 1776952320, 1141022208, 1838287872,
    1148606464, 1843841536, 67956224,   765230080,  74983424,   770226688,
    0,          423231488,  128225280,  513708032,  6653952,    425691136,
    130095104,  519772160,  1146551808, 1567424000, 1139961344, 1523084800,
    1144223232, 1560901120, 1134028288, 1521346048, 0,          33619968,
    271785984,  305274880,  268632064,  302120960,  3153920,    36773888,
    1077583360, 1111203328, 1342815744, 1376304640, 1345953280, 1379442176,
    1074445824, 1108065792,
};

}  // namespace common
}  // namespace flare
