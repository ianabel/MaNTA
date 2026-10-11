#ifndef CAPTUREDOUTPUT_HPP
#define CAPTUREDOUTPUT_HPP

// Silences the solver's diagnostic output for the duration of a scope, and
// optionally hands it back so a test can assert on it.
//
// Several tests deliberately provoke output: they run the full solver, pass a
// null pointer to ErrorChecker, or make the physics throw so static_residual has
// to report it. That output is correct and useful in production, but a passing
// test should be quiet -- otherwise real problems hide in the scroll.
//
// This redirects at the *file descriptor* level rather than swapping
// std::cout's streambuf. The project's own output is all std::print now, but
// that still lands in two different places -- std::print(stderr, ...) writes to
// the C FILE*, while std::print(ofstream, ...) goes through the stream -- and
// SUNDIALS' error handler writes to stderr from C regardless. Only the
// descriptor is common to all three.
//
// USE IT TIGHTLY. Boost.Test writes its failure messages to stdout, so an
// assertion that fires inside a captured scope is swallowed and the test fails
// silently. Capture the noisy call, restore, then assert:
//
//     std::string log;
//     {
//         CapturedOutput capture;
//         sys.runSolver(tFinal);
//         log = capture.text();
//     }
//     BOOST_TEST(log.find("...") != std::string::npos);

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

class CapturedOutput
{
public:
    CapturedOutput() { start(); }

    CapturedOutput(const CapturedOutput &) = delete;
    CapturedOutput &operator=(const CapturedOutput &) = delete;

    ~CapturedOutput()
    {
        restore();
        if (captureFd >= 0)
            ::close(captureFd);
    }

    /// Everything written to stdout or stderr since capture began.
    std::string text()
    {
        flushEverything();
        if (captureFd < 0)
            return {};

        const off_t size = ::lseek(captureFd, 0, SEEK_END);
        if (size <= 0)
            return {};

        std::string out(static_cast<size_t>(size), '\0');
        ::lseek(captureFd, 0, SEEK_SET);
        const ssize_t got = ::read(captureFd, out.data(), static_cast<size_t>(size));
        out.resize(got > 0 ? static_cast<size_t>(got) : 0);
        return out;
    }

    /// End the capture early. Idempotent; the destructor calls it too.
    void restore()
    {
        if (savedOut < 0)
            return;
        flushEverything();
        ::dup2(savedOut, STDOUT_FILENO);
        ::dup2(savedErr, STDERR_FILENO);
        ::close(savedOut);
        ::close(savedErr);
        savedOut = savedErr = -1;
    }

private:
    void start()
    {
        flushEverything();

        auto path = std::filesystem::temp_directory_path() / "manta-test-capture-XXXXXX";
        std::string templ = path.string();
        captureFd = ::mkstemp(templ.data());
        if (captureFd < 0)
            return; // capture unavailable -- let the output through rather than fail
        // Drop the directory entry immediately: the fd keeps the file alive, and
        // nothing is left behind however the test exits.
        ::unlink(templ.c_str());

        savedOut = ::dup(STDOUT_FILENO);
        savedErr = ::dup(STDERR_FILENO);
        ::dup2(captureFd, STDOUT_FILENO);
        ::dup2(captureFd, STDERR_FILENO);
    }

    static void flushEverything()
    {
        std::cout.flush();
        std::cerr.flush();
        std::fflush(stdout);
        std::fflush(stderr);
    }

    int captureFd = -1;
    int savedOut = -1;
    int savedErr = -1;
};

#endif // CAPTUREDOUTPUT_HPP
