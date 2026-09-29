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

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "master.h"
#include "grid.h"
#include "fields.h"
#include "dump.h"
#include "timeloop.h"
#include "constants.h"
#include "defines.h"

template<typename TF>
Dump<TF>::Dump(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Input& inputin):
    master(masterin), grid(gridin), fields(fieldsin),
    field3d_io(master, grid)
{
    swdump = inputin.get_item<bool>("dump", "swdump", "", false);
    swhf = false;
    do_regular = true;
    do_hf = false;

    if (swdump)
    {
        // Get the time at which the dump sections are triggered.
        sampletime = inputin.get_item<double>("dump", "sampletime", "");

        // Get list of dump variables.
        dumplist = inputin.get_list<std::string>("dump", "dumplist", "", std::vector<std::string>());
        
        // Whether to do two consecutive dumps in time
        swdoubledump = inputin.get_item<bool>("dump", "swdoubledump", "", false);
        if (swdoubledump && sampletime != inputin.get_item<double>("time", "savetime", ""))
        {
            std::string msg = "Double dump only works if sampletime is equal to savetime";
            throw std::runtime_error(msg);
        }

        swhf = inputin.get_item<bool>("dump", "swhf", "", false);
        regular_list = dumplist;
        if (swhf)
        {
            hf_sampletime = inputin.get_item<double>("dump", "hf_sampletime", "");
            hf_starttime  = inputin.get_item<double>("dump", "hf_starttime", "", 0.);
            hf_endtime    = inputin.get_item<double>("dump", "hf_endtime", "", 1.e12);
            hf_zmax       = inputin.get_item<double>("dump", "hf_zmax", "", 1.e12);
            hf_list       = inputin.get_list<std::string>("dump", "hf_dumplist", "", std::vector<std::string>());
            if (hf_list.empty())
                throw std::runtime_error("Empty high-frequency dump list");

            // The model components claim their variables from one list.
            for (auto& it : hf_list)
                if (std::find(dumplist.begin(), dumplist.end(), it) == dumplist.end())
                    dumplist.push_back(it);
        }

        // Crash on empty list.
        if (dumplist.empty())
        {
            std::string msg = "Empty Dump list";
            throw std::runtime_error(msg);
        }
    }
    else
    {
        inputin.flag_as_used("dump", "dumplist", "");
        inputin.flag_as_used("dump", "sampletime", "");
    }

}

template<typename TF>
Dump<TF>::~Dump()
{
}

template<typename TF>
void Dump<TF>::init()
{
    if (!swdump)
        return;

    isampletime = convert_to_itime(sampletime);

    if (swhf)
    {
        ihf_sampletime = convert_to_itime(hf_sampletime);
        ihf_starttime  = convert_to_itime(hf_starttime);
        ihf_endtime    = convert_to_itime(std::min(hf_endtime, 1.e9));
    }
}

template<typename TF>
void Dump<TF>::create()
{
    /* All classes (fields, thermo) have removed their dump-variables from
       dumplist by now. If it isn't empty, print warnings for invalid variables */
    if (!dumplist.empty())
    {
        for (auto& it : dumplist)
            master.print_warning("field %s in [dump][dumplist] is illegal\n", it.c_str());
    }
}

template<typename TF>
unsigned long Dump<TF>::get_time_limit(unsigned long itime)
{
    if (!swdump)
        return Constants::ulhuge;

    unsigned long limit = isampletime - itime % isampletime;

    if (swhf && itime < ihf_endtime)
    {
        // First sample time after itime that lies in the window.
        unsigned long next = itime + ihf_sampletime - itime % ihf_sampletime;
        if (next < ihf_starttime)
            next = ihf_starttime + (ihf_sampletime - ihf_starttime % ihf_sampletime) % ihf_sampletime;
        if (next <= ihf_endtime)
            limit = std::min(limit, next - itime);
    }

    return limit;
}

template<typename TF>
bool Dump<TF>::do_dump(unsigned long itime, unsigned long idt)
{
    // Check if dump is enabled.
    if (!swdump)
        return false;

    do_regular = (itime % isampletime == 0) || (swdoubledump && ((itime + idt) % isampletime == 0));
    do_hf = swhf && (itime % ihf_sampletime == 0) && (itime >= ihf_starttime) && (itime <= ihf_endtime);

    return do_regular || do_hf;
}

template<typename TF>
bool Dump<TF>::needs(const std::string& varname)
{
    const bool in_regular = std::find(regular_list.begin(), regular_list.end(), varname) != regular_list.end();
    const bool in_hf = std::find(hf_list.begin(), hf_list.end(), varname) != hf_list.end();
    return (do_regular && in_regular) || (do_hf && in_hf);
}

template<typename TF>
std::vector<std::string>& Dump<TF>::get_dumplist()
{
    return dumplist;
}

template<typename TF>
void Dump<TF>::save_dump(TF* data, const std::string& varname, int iotime)
{
    auto& gd = grid.get_grid_data();
    const double no_offset = 0.;
    char filename[256];

    const bool in_regular = std::find(regular_list.begin(), regular_list.end(), varname) != regular_list.end();
    const bool in_hf = std::find(hf_list.begin(), hf_list.end(), varname) != hf_list.end();

    if (do_hf && in_hf)
    {
        int kend_hf = gd.kstart;
        while (kend_hf < gd.kend && gd.z[kend_hf] <= hf_zmax)
            ++kend_hf;

        std::snprintf(filename, 256, "%s_hf.%07d", varname.c_str(), iotime);
        std::ifstream infile_hf(filename);

        if (infile_hf.good())
            master.print_message("%s already exists\n", filename);
        else if (field3d_io.save_field3d_float(data, filename, gd.kstart, kend_hf))
        {
            master.print_message("Saving \"%s\" ... FAILED\n", filename);
            throw std::runtime_error("Writing error in dump");
        }
    }

    if (!(do_regular && in_regular))
        return;

    std::snprintf(filename, 256, "%s.%07d", varname.c_str(), iotime);
    std::ifstream infile(filename);

    if (infile.good())
    {
        master.print_message("%s already exists\n", filename);
    }
    else
    {

        auto tmp1 = fields.get_tmp();
        auto tmp2 = fields.get_tmp();

        if (field3d_io.save_field3d(
                    data,
                    tmp1->fld.data(), tmp2->fld.data(),
                    filename, no_offset,
                    gd.kstart, gd.kend))
        {
            master.print_message("Saving \"%s\" ... FAILED\n", filename);
            throw std::runtime_error("Writing error in dump");
        }

        fields.release_tmp(tmp1);
        fields.release_tmp(tmp2);
    }
}


#ifdef FLOAT_SINGLE
template class Dump<float>;
#else
template class Dump<double>;
#endif
