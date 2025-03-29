////////////////////////////////////////////////////////////////////////////////
//  Parallel Program Induction (PPI) is free software: you can redistribute it
//  and/or modify it under the terms of the GNU General Public License as
//  published by the Free Software Foundation, either version 3 of the License,
//  or (at your option) any later version.
//
//  PPI is distributed in the hope that it will be useful, but WITHOUT ANY
//  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
//  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
//  details.
//
//  You should have received a copy of the GNU General Public License along
//  with PPI.  If not, see <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////////////////

#ifndef __client_h
#define __client_h

#include "Poco/Net/StreamSocket.h"
#include "Poco/Net/SocketAddress.h"
#include "Poco/Thread.h"
#include "Poco/Timespan.h"

using Poco::Net::StreamSocket;
using Poco::Net::Socket;
using Poco::Net::SocketAddress;
using Poco::Thread;

#include "../common/common.h"

/******************************************************************************/
class Client: public Common, public Poco::Runnable {
public:
   Client( StreamSocket& s, const char* server, const std::string& results, bool &isrunning, Poco::FastMutex &mutex ):
      Common( s ), m_server( server ), m_results( results ), m_isrunning(isrunning), m_mutex(mutex) {}

   int  Connect();
   void Disconnect();
   void SndIndividual();

   virtual void run()
   {
      try {
         SndIndividual();
      } catch (Poco::Exception& exc) {
         std::cerr << "SndIndividual(): " << exc.displayText() << std::endl;
      } catch (...) {
         std::cerr << "SndIndividual(): Unknown error!" << std::endl;
      }

      // Ensures that this thread will release m_isrunning whatever happens!
      {
         Poco::FastMutex::ScopedLock lock(m_mutex);
         m_isrunning = 0;
      }
   }

public:
   static double time;

private:
   const char* m_server;
   const std::string m_results;
   bool &m_isrunning;
   Poco::FastMutex &m_mutex;
};

/******************************************************************************/
#endif
