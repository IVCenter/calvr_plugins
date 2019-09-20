#ifndef INTERACTABLE_H
#define INTERACTABLE_H

#include <cvrKernel/InteractionEvent.h>

class Interactable
{
public:
	Interactable() {}
	virtual ~Interactable() {}

	virtual bool processEvent(cvr::InteractionEvent* e) = 0;
	virtual unsigned int getPriority() const { return _priority; }
	virtual void setPriority(unsigned int p) { _priority = p; }
	virtual bool operator <(const Interactable &b) const
	{
		return getPriority() < b.getPriority();
	}

protected:
	unsigned int _priority;
};

#endif // !INTERACTABLE_H
