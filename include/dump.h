/*
 * MicroHH
 * Copyright (c) 2011-2024 Chiel van Heerwaarden
 * Copyright (c) 2011-2024 Thijs Heus
 * Copyright (c) 2014-2024 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DUMP_H
#define DUMP_H

class Master;
class Input;
template<typename> class Grid;
template<typename> class Fields;

template<typename TF>
class Dump
{
    public:
        Dump(Master&, Grid<TF>&, Fields<TF>&, Input&);
        ~Dump();

        void init();
        void create();

        unsigned long get_time_limit(unsigned long);
        bool get_switch() { return swdump; }
        std::vector<std::string>& get_dumplist();

        bool do_dump(unsigned long, unsigned long);
        bool needs(const std::string&);
        void save_dump(TF*, const std::string&, int);

    private:
        Master& master;
        Grid<TF>& grid;
        Fields<TF>& fields;
        Field3d_io<TF> field3d_io;

        std::vector<std::string> dumplist; // List with all dumps from the ini file.
        bool swdump;                       // Statistics on/off switch
        bool swdoubledump;                 // On/off switch for two consecutive dumps in time
        double sampletime;
        unsigned long isampletime;

        // Second, high-frequency stream: own cadence and time window, float32, levels below hf_zmax.
        bool swhf;
        std::vector<std::string> regular_list;
        std::vector<std::string> hf_list;
        double hf_sampletime;
        double hf_starttime;
        double hf_endtime;
        double hf_zmax;
        unsigned long ihf_sampletime;
        unsigned long ihf_starttime;
        unsigned long ihf_endtime;
        bool do_regular;
        bool do_hf;
};
#endif
